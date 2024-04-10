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
import zipfile
import io
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from bpe_exporter import read_spm_model, Token
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
        self._write(ctx.with_subname("out_proj.weight").with_quant(Quant.NONE), model.output_layer.weight)

    def export_linear(self, ctx: Context, module: nn.Linear):
        self._write(ctx.with_subname("weight"), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)

    def export_self_attn(self, ctx, attn):
        self.export_linear(ctx.with_subname("qkv_proj"), attn.query_key_value)
        self._write(ctx.with_subname("out_proj.weight"), attn.dense.weight)

    def export_mlp(self, ctx, mlp):
        self._write(ctx.with_subname("dense1.weight"), mlp.dense_h_to_4h.weight)
        self._write(ctx.with_subname("dense2.weight"), mlp.dense_4h_to_h.weight)

    def export_glm_block(self, ctx, block):
        self.export_rms_norm(ctx.with_subname("norm"), block.input_layernorm)
        self.export_rms_norm(ctx.with_subname("attn_norm"), block.post_attention_layernorm)
        self.export_self_attn(ctx.with_subname("attn"), block.self_attention)
        self.export_mlp(ctx.with_subname("mlp"), block.mlp)

    def export_rms_norm(self, ctx, module):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)

    @classmethod
    def generate_config(cls, chatglm_config, model_name) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config[model_name] = {}
        
        section = config[model_name]
        section["hidden_size"] = str(chatglm_config.hidden_size)
        section["vocab_size"] = str(chatglm_config.padded_vocab_size)
        section["kv_channels"] = str(chatglm_config.kv_channels)
        section["seq_length"] = str(chatglm_config.seq_length)
        section["hidden_size_per_attention_head"] = str(chatglm_config.kv_channels)
        section["multi_query_group_num"] = str(chatglm_config.multi_query_group_num)
        section["norm_eps"] = str(chatglm_config.layernorm_epsilon)
        section["ffn_hidden_size"] = str(chatglm_config.ffn_hidden_size)
        section["num_layers"] = str(chatglm_config.num_layers)
        section["symbol_gmask"] = str(64790)
        section["symbol_sop"] = str(64792)
        section["symbol_eos"] = str(2)

        return config

    @classmethod
    def export(cls, chatglm_model, fp, quant: Quant) -> configparser.ConfigParser:
        model = chatglm_model.base_model
        config = chatglm_model.config

        ctx = Context("chatglm3", quant=quant)
        with TensorWriter(fp) as writer:
            exporter = ChatGLMExporter(writer)
            exporter.export_chatglm(ctx, model)

        ini_config = cls.generate_config(config, "chatglm")
        ini_config["model"] = {}
        ini_config["model"]["type"] = "chatglm3"
        ini_config["model"]["model_file"] = path.basename(MODEL_BIN)

        return ini_config

MODEL_BIN = "model.bin"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    import argparse

    parser = argparse.ArgumentParser(description='export ChatGLM model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the ChatGLM model name in huggingface.', default="THUDM/chatglm3-6b")
    parser.add_argument('-quantization', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output', type=str, help='output model package.', default="chatglm.llmpkg")
    args = parser.parse_args()
    
    huggingface_name = args.huggingface_name
    model_type = "chatglm3"
    quant = args.quantization

    model = AutoModel.from_pretrained(huggingface_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_name, trust_remote_code=True)

    model = model.eval()

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = ChatGLMExporter.export(model, fp, args.quantization)

        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        libllm_tokenizer = read_spm_model(args.huggingface_name)
        special_tokens = tokenizer.tokenizer.special_tokens
        special_tokens = list(special_tokens.items())
        special_tokens.sort(key=lambda t: t[1])
        for name, token_id in special_tokens:
            print(f"add control token: {token_id} {name}")
            libllm_tokenizer.add_token(Token(token_id, Token.FLAG_CONTROL, piece_display=name, weight=-100.0))
        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            libllm_tokenizer.save(fp)
        
        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            libllm_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))
