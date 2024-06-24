# The MIT License (MIT)
#
# Copyright (c) 2024 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
import io
import torch
import configparser
import sys
import zipfile
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from bpe_exporter import read_spm_model, Token
from llama_exporter import LlamaExporter
from torch import nn
from transformers import AutoTokenizer, pipeline


class BilibiliIndexExporter(LlamaExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export(self, ctx: Context, model):
        base_model = model.base_model

        self.export_embedding(ctx.with_subname("embd"), base_model.embed_tokens)
        self._export_rms_norm(ctx.with_subname("norm"), base_model.norm)
        self._export_rope(ctx.with_subname("rope"), base_model.layers[0].self_attn.rotary_emb)
        for idx, block in enumerate(base_model.layers):
            self._export_block(ctx.with_subname("block" + str(idx)), block)
        self._write(ctx.with_subname("out_proj.weight"), torch.nn.functional.normalize(model.lm_head.weight))

    @classmethod
    def generate_config(cls, config) -> configparser.ConfigParser:
        ini_config = configparser.ConfigParser()
        ini_config["index"] = {}

        assert config.hidden_act == "silu"
        assert config.norm_head in {1, True}
        
        section = ini_config["index"]
        section["hidden_size"] = str(config.hidden_size)
        section["num_heads"] = str(config.num_attention_heads)
        section["num_key_value_heads"] = str(config.num_key_value_heads)
        section["intermediate_size"] = str(config.intermediate_size)
        section["norm_eps"] = str(config.rms_norm_eps)
        section["num_layers"] = str(config.num_hidden_layers)
        section["vocab_size"] = str(config.vocab_size)
        section["max_ctx_length"] = str(config.max_position_embeddings)
        section["bot_token_id"] = "1"
        section["eot_token_id"] = "2"

        return ini_config

    @classmethod
    def export(cls, bili_model, fp, quant: Quant):
        model = bili_model
        config = bili_model.config

        ctx = Context("index", quant=quant)
        with TensorWriter(fp) as writer:
            exporter = BilibiliIndexExporter(writer)
            exporter._export(ctx, model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "index"
        ini_config["model"]["model_file"] = path.basename(MODEL_BIN)

        return ini_config

def run_bilibili_index(huggingface_name):
    tokenizer = AutoTokenizer.from_pretrained(huggingface_name, trust_remote_code=True)
    generator = pipeline("text-generation",
                        model=huggingface_name,
                        tokenizer=tokenizer, trust_remote_code=True, 
                        device="cpu")


    query = "你好"
    model_input = []
    model_input.append({"role": "user", "content": query})

    model_output = generator(model_input, max_new_tokens=300, top_k=5, top_p=0.8, temperature=0.3, repetition_penalty=1.1, do_sample=True)

    print('User:', query)
    print('Model:', model_output)

MODEL_NAME = "IndexTeam/Index-1.9B-Character"
MODEL_BIN = "model.bin"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description='export llama model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the llama model name in huggingface.', default=MODEL_NAME)
    parser.add_argument('-quant', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output', type=str, help='output file name.', default="bilibili_index.llmpkg")
    parser.add_argument('-run', action="store_true")
    args = parser.parse_args()

    if args.run:
        run_bilibili_index(args.huggingface_name)
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.huggingface_name, device_map="auto", trust_remote_code=True).eval()
    model.to("cpu")

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = BilibiliIndexExporter.export(model, fp, args.quant)

        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        libllm_tokenizer = read_spm_model(args.huggingface_name)
        libllm_tokenizer._vocab[3] = Token(3, Token.FLAG_CONTROL, piece_display="<|reserved_0|>", weight=-100.0)
        libllm_tokenizer._vocab[4] = Token(4, Token.FLAG_CONTROL, piece_display="<|reserved_1|>", weight=-100.0)
        libllm_tokenizer.config.add_prefix_space = False
        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            libllm_tokenizer.save(fp)
        
        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            libllm_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))

