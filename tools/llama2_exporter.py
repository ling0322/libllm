# The MIT License (MIT)
#
# Copyright (c) 2023 Xiaoyang Chen
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
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from spm_exporter import TokenizerExporter
from torch import nn

class Llama2Exporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export_llama2(self, ctx: Context, model):
        base_model = model.base_model

        self.export_embedding(ctx.with_subname("embd"), base_model.embed_tokens)
        self._export_rms_norm(ctx.with_subname("norm"), base_model.norm)
        self._export_rope(ctx.with_subname("rope"), base_model.layers[0].self_attn.rotary_emb)
        for idx, block in enumerate(base_model.layers):
            self._export_block(ctx.with_subname("block" + str(idx)), block)
        self._write(ctx.with_subname("out_weight"), model.lm_head.weight)

    def _export_rms_norm(self, ctx: Context, module):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)

    def _export_block(self, ctx: Context, model_block):
        self._export_rms_norm(ctx.with_subname("input_norm"), model_block.input_layernorm)
        self._export_attn(ctx.with_subname("attn"), model_block.self_attn)
        self._export_rms_norm(ctx.with_subname("post_attn_norm"), model_block.post_attention_layernorm)
        self._export_mlp(ctx.with_subname("mlp"), model_block.mlp)

    def _export_rope(self, ctx: Context, rope_embd):
        cos_cached = torch.squeeze(rope_embd.cos_cached)
        sin_cached = torch.squeeze(rope_embd.sin_cached)

        rope = torch.stack((cos_cached, sin_cached))
        self._write(ctx.with_quant(Quant.NONE), rope)

    def _export_attn(self, ctx: Context, attn_block):
        q_proj = attn_block.q_proj.weight
        k_proj = attn_block.k_proj.weight
        v_proj = attn_block.v_proj.weight

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0)
        self._write(ctx.with_subname("qkv_proj"), qkv_proj)
        self._write(ctx.with_subname("out_proj"), attn_block.o_proj.weight)

    def _export_mlp(self, ctx: Context, attn_block):
        w_gate = attn_block.gate_proj.weight
        w_up = attn_block.up_proj.weight
        w_gate_up = torch.cat((w_gate, w_up), dim=0)
        self._write(ctx.with_subname("gate_up_proj"), w_gate_up)
        self._write(ctx.with_subname("down_proj"), attn_block.down_proj.weight)

    @classmethod
    def generate_config(cls, llama2_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["llama2"] = {}

        assert llama2_config.num_key_value_heads == llama2_config.num_attention_heads
        assert llama2_config.rope_scaling is None
        assert llama2_config.pretraining_tp == 1
        assert llama2_config.hidden_act == "silu"
        
        chatglm2_section = config["llama2"]
        chatglm2_section["hidden_size"] = str(llama2_config.hidden_size)
        chatglm2_section["num_heads"] = str(llama2_config.num_attention_heads)
        chatglm2_section["intermediate_size"] = str(llama2_config.intermediate_size)
        chatglm2_section["norm_eps"] = str(llama2_config.rms_norm_eps)
        chatglm2_section["num_layers"] = str(llama2_config.num_hidden_layers)
        chatglm2_section["vocab_size"] = str(llama2_config.vocab_size)
        chatglm2_section["max_ctx_length"] = str(llama2_config.max_position_embeddings)
        chatglm2_section["bos_token_id"] = "1"
        chatglm2_section["eos_token_id"] = "2"

        return config

    @classmethod
    def export(cls, llama2_model, output_prefix: str, quant: Quant):
        model = llama2_model
        config = llama2_model.config

        quant_name = "fp32"
        if quant != Quant.NONE:
            quant_name = str(quant.name).lower()

        ctx = Context("llama", quant=quant)
        bin_filename = f"{output_prefix}.{quant_name}.bin"
        with TensorWriter(bin_filename) as writer:
            exporter = Llama2Exporter(writer)
            exporter._export_llama2(ctx, model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "llama"
        ini_config["model"]["model_file"] = path.basename(bin_filename)

        return ini_config


MODEL_NAME = "meta-llama/Llama-2-7b-hf"

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from transformers.models.llama import LlamaForCausalLM


    parser = argparse.ArgumentParser(description='export llama model from huggingface to libllm format.')
    parser.add_argument('--name', type=str, help='the llama model name in huggingface.', default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--quantization', type=Quant.parse, help='quantization type, "q4" or "q4sym" or "none"', default=Quant.Q4)
    parser.add_argument('--output', type=str, help='output file name.', default="llama2")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.name, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(args.name, trust_remote_code=True)
    model = model.eval()

    output_prefix = args.output
    config = Llama2Exporter.export(model, output_prefix, args.quantization)
    tokenizer_conifg = TokenizerExporter.export(args.name, output_prefix)
    tokenizer_conifg.update_config(config, "tokenizer")

    with open(output_prefix + ".config", "w") as fp:
        config.write(fp)
