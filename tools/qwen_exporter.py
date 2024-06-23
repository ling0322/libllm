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
from bpe_exporter import read_transformers_fast_bpe_model
from llama_exporter import LlamaExporter
from torch import nn

class Qwen2Exporter(LlamaExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export_attn(self, ctx: Context, attn_block):
        """llama model didn't have bias in qkv_proj but qwen does."""
        q_proj = attn_block.q_proj.weight
        k_proj = attn_block.k_proj.weight
        v_proj = attn_block.v_proj.weight

        q_proj_bias = attn_block.q_proj.bias
        k_proj_bias = attn_block.k_proj.bias
        v_proj_bias = attn_block.v_proj.bias

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0)
        qkv_proj_bias = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0)
        self._write(ctx.with_subname("qkv_proj.weight"), qkv_proj)
        self._write(ctx.with_subname("qkv_proj.bias").with_quant(Quant.NONE), qkv_proj_bias)
        self._write(ctx.with_subname("out_proj.weight"), attn_block.o_proj.weight)

    @classmethod
    def generate_config(cls, qwen_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["qwen"] = {}

        assert qwen_config.hidden_act == "silu"
        
        qwen_section = config["qwen"]
        qwen_section["hidden_size"] = str(qwen_config.hidden_size)
        qwen_section["num_heads"] = str(qwen_config.num_attention_heads)
        qwen_section["num_key_value_heads"] = str(qwen_config.num_key_value_heads)
        qwen_section["intermediate_size"] = str(qwen_config.intermediate_size)
        qwen_section["norm_eps"] = str(qwen_config.rms_norm_eps)
        qwen_section["num_layers"] = str(qwen_config.num_hidden_layers)
        qwen_section["vocab_size"] = str(qwen_config.vocab_size)
        qwen_section["max_ctx_length"] = str(qwen_config.max_position_embeddings)
        qwen_section["qkv_proj_bias"] = "true"

        return config

    @classmethod
    def export(cls, qwen_model, fp, quant: Quant):
        model = qwen_model
        config = qwen_model.config

        ctx = Context("qwen", quant=quant)
        with TensorWriter(fp) as writer:
            exporter = Qwen2Exporter(writer)
            exporter._export(ctx, model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "qwen"
        ini_config["model"]["model_file"] = path.basename(MODEL_BIN)

        return ini_config

def run_qwen(huggingface_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(huggingface_name, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(huggingface_name)

    prompt = "hi"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = '<|im_start|> system\nYou are a helpful assistant.<|im_end|> \n<|im_start|> user\nhi<|im_end|> \n<|im_start|> assistant\n'
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
MODEL_BIN = "model.bin"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description='export llama model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the llama model name in huggingface.', default=MODEL_NAME)
    parser.add_argument('-quantization', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output', type=str, help='output file name.', default="qwen.llmpkg")
    parser.add_argument('-run', action="store_true")
    args = parser.parse_args()

    if args.run:
        run_qwen(args.huggingface_name)
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.huggingface_name, device_map="auto").eval()
    model.to("cpu")

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = Qwen2Exporter.export(model, fp, args.quantization)

        eot_token_id = tokenizer.added_tokens_encoder["<|endoftext|>"]
        im_start_token_id = tokenizer.added_tokens_encoder["<|im_start|>"]
        im_end_token_id = tokenizer.added_tokens_encoder["<|im_end|>"]
        config["qwen"]["eot_token_id"] = str(eot_token_id)
        config["qwen"]["im_start_token_id"] = str(im_start_token_id)
        config["qwen"]["im_end_token_id"] = str(im_end_token_id)
        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        libllm_tokenizer = read_transformers_fast_bpe_model(args.huggingface_name)
        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            libllm_tokenizer.save(fp)
        
        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            libllm_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))

