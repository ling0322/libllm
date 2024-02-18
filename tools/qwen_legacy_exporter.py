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
import torch
import configparser
import sys
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from bpe_exporter import read_tiktoken_model
from torch import nn

class QwenExporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export_qwen(self, ctx: Context, model):
        base_model = model.base_model
        max_length = model.config.max_position_embeddings
        base_model.rotary_emb.update_rotary_pos_emb_cache(max_length)
        self._export_rope(ctx.with_subname("rope"), max_length, base_model.rotary_emb)

        self.export_embedding(ctx.with_subname("embd"), base_model.wte)
        self._export_rms_norm(ctx.with_subname("norm"), base_model.ln_f)
        for idx, block in enumerate(base_model.h):
            self._export_block(ctx.with_subname("block" + str(idx)), block)
        self._write(ctx.with_subname("out_weight").with_quant(Quant.NONE), model.lm_head.weight)

    def _export_rms_norm(self, ctx: Context, module):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)

    def _export_block(self, ctx: Context, model_block):
        self._export_rms_norm(ctx.with_subname("input_norm"), model_block.ln_1)
        self._export_attn(ctx.with_subname("attn"), model_block.attn)
        self._export_rms_norm(ctx.with_subname("post_attn_norm"), model_block.ln_2)
        self._export_mlp(ctx.with_subname("mlp"), model_block.mlp)

    def _export_rope(self, ctx: Context, max_length: int, rope_embd):
        cos, sin = rope_embd._rotary_pos_emb_cache
        cos_cached = torch.squeeze(cos)
        sin_cached = torch.squeeze(sin)

        rope = torch.stack((cos_cached, sin_cached))
        rope = rope[:, :max_length, :]
        self._write(ctx.with_quant(Quant.NONE), rope)

    def _export_attn(self, ctx: Context, attn_block):
        self._write(ctx.with_subname("qkv_proj"), attn_block.c_attn.weight)
        self._write(ctx.with_subname("qkv_proj_bias").with_quant(Quant.NONE), attn_block.c_attn.bias)
        self._write(ctx.with_subname("out_proj"), attn_block.c_proj.weight)

    def _export_mlp(self, ctx: Context, attn_block):
        w_gate = attn_block.w2.weight
        w_up = attn_block.w1.weight
        w_gate_up = torch.cat((w_gate, w_up), dim=0)
        self._write(ctx.with_subname("gate_up_proj"), w_gate_up)
        self._write(ctx.with_subname("down_proj"), attn_block.c_proj.weight)

    @classmethod
    def generate_config(cls, qwen_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["qwen"] = {}

        assert qwen_config.rotary_pct == 1.0
        assert qwen_config.no_bias == True
        
        qwen_section = config["qwen"]
        qwen_section["hidden_size"] = str(qwen_config.hidden_size)
        qwen_section["num_heads"] = str(qwen_config.num_attention_heads)
        qwen_section["intermediate_size"] = str(qwen_config.intermediate_size // 2)
        qwen_section["norm_eps"] = str(qwen_config.layer_norm_epsilon)
        qwen_section["num_layers"] = str(qwen_config.num_hidden_layers)
        qwen_section["vocab_size"] = str(qwen_config.vocab_size)
        qwen_section["max_ctx_length"] = str(qwen_config.max_position_embeddings)
        qwen_section["qkv_proj_bias"] = "true"

        return config

    @classmethod
    def export(cls, llama2_model, output_prefix: str, quant: Quant):
        model = llama2_model
        config = llama2_model.config

        quant_name = "fp32"
        if quant != Quant.NONE:
            quant_name = str(quant.name).lower()

        ctx = Context("qwen", quant=quant)
        bin_filename = f"{output_prefix}.{quant_name}.bin"
        with TensorWriter(bin_filename) as writer:
            exporter = QwenExporter(writer)
            exporter._export_qwen(ctx, model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "qwen"
        ini_config["model"]["model_file"] = path.basename(bin_filename)

        return ini_config

def run_qwen(huggingface_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(huggingface_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(huggingface_name, device_map="auto", trust_remote_code=True).eval()
    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)

MODEL_NAME = "Qwen/Qwen-1_8B-Chat"

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description='export llama model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the llama model name in huggingface.', default=MODEL_NAME)
    parser.add_argument('-quantization', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output', type=str, help='output file name.', default="qwen")
    parser.add_argument('-run', action="store_true")
    args = parser.parse_args()

    if args.run:
        run_qwen(args.huggingface_name)
        sys.exit(0)

    output_prefix = args.output
    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.huggingface_name, device_map="auto", trust_remote_code=True).eval()
    model.to("cpu")

    config = QwenExporter.export(model, output_prefix, args.quantization)
    eot_token_id = tokenizer.special_tokens["<|endoftext|>"]
    im_start_token_id = tokenizer.special_tokens["<|im_start|>"]
    im_end_token_id = tokenizer.special_tokens["<|im_end|>"]
    config["qwen"]["eot_token_id"] = str(eot_token_id)
    config["qwen"]["im_start_token_id"] = str(im_start_token_id)
    config["qwen"]["im_end_token_id"] = str(im_end_token_id)


    libllm_tokenizer = read_tiktoken_model(tokenizer.tokenizer)
    libllm_tokenizer.save(output_prefix + ".tokenizer.bin")
    libllm_tokenizer.get_config().update_config(config, "tokenizer")

    with open(output_prefix + ".config", "w") as fp:
        config.write(fp)
