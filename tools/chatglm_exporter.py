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
import sys
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from spm_exporter import read_spm_model, Token
from torch import nn

class ChatGLMExporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def export_chatglm_embedding(self, ctx, chatglm2_embd):
        self.export_embedding(ctx, chatglm2_embd.word_embeddings)

    def export_rope(self, ctx: Context, model):
        rope = model.rotary_pos_emb(model.seq_length)
        self._write(ctx.with_subname("rope").with_quant(Quant.NONE), rope)

    def export_chatglm(self, ctx: Context, model):
        self.export_chatglm_embedding(ctx.with_subname("embd"), model.embedding)
        self.export_rope(ctx, model)
        for idx, layer in enumerate(model.encoder.layers):
            self.export_glm_block(ctx.with_subname(f"block{idx}"), layer)
        self.export_rms_norm(ctx.with_subname("final_norm"), model.encoder.final_layernorm)
        self._write(ctx.with_subname("output_weight").with_quant(Quant.NONE), model.output_layer.weight)

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
    def generate_config(cls, chatglm2_config, model_name) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config[model_name] = {}
        
        chatglm2_section = config[model_name]
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
    def export(cls, chatglm_model, output_prefix: str, quant: Quant) -> configparser.ConfigParser:
        model = chatglm_model.base_model
        config = chatglm_model.config

        assert config.rmsnorm == True
        assert config.add_qkv_bias == True
        assert config.add_bias_linear == False
        assert config.post_layer_norm == True
        assert config.apply_residual_connection_post_layernorm == False

        quant_name = "fp32"
        if quant != Quant.NONE:
            quant_name = str(quant.name).lower()

        ctx = Context("chatglm", quant=quant)
        bin_filename = f"{output_prefix}.{quant_name}.bin"
        with TensorWriter(bin_filename) as writer:
            exporter = ChatGLMExporter(writer)
            exporter.export_chatglm(ctx, model)        

        ini_config = cls.generate_config(config, model_name)
        ini_config["model"] = {}
        ini_config["model"]["model_file"] = path.basename(bin_filename)

        return ini_config

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    import argparse

    parser = argparse.ArgumentParser(description='export ChatGLM model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the ChatGLM model name in huggingface.', default="THUDM/chatglm3-6b")
    parser.add_argument('-chatglm_version', type=int, help='version of the ChatGLM model, for example, 3 for chatglm3.', default=0)
    parser.add_argument('-quantization', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output_dir', type=str, help='output directory name, files will be written to this directory with prefix <name>.', default=".")
    args = parser.parse_args()
    
    huggingface_name = args.huggingface_name
    model_name = "chatglm"
    model_version = args.chatglm_version
    if model_version == 0:
        if huggingface_name[: 6] == "THUDM/":
            name = huggingface_name[6: ].split("-")[0]
            if name == "chatglm2":
                model_version = 2
            elif name == "chatglm3":
                model_version = 3
    if model_version == 0:
        print("Unable to infer model version from huggingface_name, please specify it by -name. For example, -name chatglm3-6b")
        sys.exit(1)

    model_type = model_name + str(model_version)
    output_prefix = path.join(args.output_dir, model_type)
    quant = args.quantization

    model = AutoModel.from_pretrained(huggingface_name, trust_remote_code=True)
    model = model.eval()

    config = ChatGLMExporter.export(model, output_prefix, quant)
    config["model"]["type"] = model_type
    lytok_model = read_spm_model(huggingface_name)
    if model_version == 3:
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        special_tokens = tokenizer.tokenizer.special_tokens
        special_tokens = list(special_tokens.items())
        special_tokens.sort(key=lambda t: t[1])
        for name, token_id in special_tokens:
            print(f"add control token: {token_id} {name}")
            lytok_model.add_token(Token(token_id, Token.FLAG_CONTROL, piece_display=name, weight=-100.0))

    lytok_model.save(output_prefix + ".tokenizer.bin")
    lytok_model.get_config().update_config(config, "tokenizer")

    with open(output_prefix + ".config", "w") as fp:
        config.write(fp)
