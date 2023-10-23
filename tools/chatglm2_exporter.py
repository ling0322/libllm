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

import configparser
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from spm_exporter import TokenizerExporter
from torch import nn

class ChatGLM2Exporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def export_chatglm2_embedding(self, ctx, chatglm2_embd):
        self.export_embedding(ctx, chatglm2_embd.word_embeddings)

    def export_rope(self, ctx: Context, model):
        rope = model.rotary_pos_emb(model.seq_length)
        self._write(ctx.with_subname("rope").with_quant(Quant.NONE), rope)

    def export_chatglm2(self, ctx: Context, model):
        self.export_chatglm2_embedding(ctx.with_subname("embd"), model.embedding)
        self.export_rope(ctx, model)
        for idx, layer in enumerate(model.encoder.layers):
            self.export_glm_block(ctx.with_subname(f"block{idx}"), layer)
        self.export_rms_norm(ctx.with_subname("final_norm"), model.encoder.final_layernorm)
        self._write(ctx.with_subname("output_weight"), model.output_layer.weight)

    def export_linear(self, ctx: Context, module: nn.Linear):
        self._write(ctx.with_subname("weight"), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)

    def export_self_attn(self, ctx, attn):
        self.export_linear(ctx.with_subname("qkv_proj"), attn.query_key_value)
        self._write(ctx.with_subname("dense_weight"), attn.dense.weight)

    def export_mlp(self, ctx, mlp):
        self._write(ctx.with_subname("dense1_weight"), mlp.dense_h_to_4h.weight)
        self._write(ctx.with_subname("dense2_weight"), mlp.dense_4h_to_h.weight)

    def export_glm_block(self, ctx, block):
        self.export_rms_norm(ctx.with_subname("norm"), block.input_layernorm)
        self.export_rms_norm(ctx.with_subname("attn_norm"), block.post_attention_layernorm)
        self.export_self_attn(ctx.with_subname("attn"), block.self_attention)
        self.export_mlp(ctx.with_subname("mlp"), block.mlp)

    def export_rms_norm(self, ctx, module):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)

    @classmethod
    def generate_config(cls, chatglm2_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["chatglm2"] = {}
        
        chatglm2_section = config["chatglm2"]
        chatglm2_section["hidden_size"] = str(chatglm2_config.hidden_size)
        chatglm2_section["vocab_size"] = str(chatglm2_config.padded_vocab_size)
        chatglm2_section["kv_channels"] = str(chatglm2_config.kv_channels)
        chatglm2_section["seq_length"] = str(chatglm2_config.seq_length)
        chatglm2_section["hidden_size_per_attention_head"] = str(chatglm2_config.kv_channels)
        chatglm2_section["multi_query_group_num"] = str(chatglm2_config.multi_query_group_num)
        chatglm2_section["norm_eps"] = str(chatglm2_config.layernorm_epsilon)
        chatglm2_section["ffn_hidden_size"] = str(chatglm2_config.ffn_hidden_size)
        chatglm2_section["num_layers"] = str(chatglm2_config.num_layers)
        chatglm2_section["symbol_gmask"] = str(64790)
        chatglm2_section["symbol_sop"] = str(64792)
        chatglm2_section["symbol_eos"] = str(2)

        return config

    @classmethod
    def export(cls, chatglm2_model, output_prefix: str, quant: Quant) -> configparser.ConfigParser:
        model = chatglm2_model.base_model
        config = chatglm2_model.config

        assert config.rmsnorm == True
        assert config.add_qkv_bias == True
        assert config.add_bias_linear == False
        assert config.post_layer_norm == True
        assert config.apply_residual_connection_post_layernorm == False

        quant_name = "fp32"
        if quant != Quant.NONE:
            quant_name = str(quant.name).lower()

        ctx = Context("chatglm2", quant=quant)
        bin_filename = f"{output_prefix}.{quant_name}.bin"
        with TensorWriter(bin_filename) as writer:
            exporter = ChatGLM2Exporter(writer)
            exporter.export_chatglm2(ctx, model)        

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "chatglm2"
        ini_config["model"]["model_file"] = path.basename(bin_filename)

        return ini_config

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    import argparse

    parser = argparse.ArgumentParser(description='export ChatGLM2 model from huggingface to libllm format.')
    parser.add_argument('--name', type=str, help='the ChatGLM2 model name in huggingface.', default="THUDM/chatglm2-6b")
    parser.add_argument('--quantization', type=Quant.parse, help='quantization type, "q4" or "q4sym" or "none"', default=Quant.Q4)
    parser.add_argument('--output', type=str, help='output file name.', default="chatglm2")
    args = parser.parse_args()
    
    model_name = args.name
    output_prefix = args.output
    quant = args.quantization

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval()

    config = ChatGLM2Exporter.export(model, output_prefix, quant)
    tokenizer_conifg = TokenizerExporter.export(model_name, output_prefix)
    tokenizer_conifg.update_config(config, "tokenizer")

    with open(output_prefix + ".config", "w") as fp:
        config.write(fp)
