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
import tempfile
import configparser
import zipfile
import io
import sys
import os
import urllib
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from bpe_exporter import read_spm_model, read_transformers_fast_bpe_model

class WhisperExporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export_conv1d(self, ctx: Context, module):
        self._write(ctx.with_subname("weight"), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)
        assert(module.dilation == (1, ))
        assert(module.padding == (1, ))

    def _export_enc_pos_embd(self, ctx: Context, module):
        self._write(ctx, module.weight)

    def _export_attn(self, ctx: Context, module, cross_attn=False):
        q_proj = module.q_proj.weight
        k_proj = module.k_proj.weight
        v_proj = module.v_proj.weight

        if cross_attn:
            kv_proj = torch.cat((k_proj, v_proj), dim=0)
            self._write(ctx.with_subname("kv_proj.weight"), kv_proj)
            self._write(ctx.with_subname("q_proj.weight"), q_proj)
        else:
            qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0)
            self._write(ctx.with_subname("qkv_proj.weight"), qkv_proj)

        assert module.q_proj.bias is not None
        assert module.v_proj.bias is not None
        assert module.k_proj.bias is None
        q_bias = module.q_proj.bias
        k_bias = torch.zeros_like(q_bias)
        v_bias = module.v_proj.bias

        if cross_attn:
            kv_bias = torch.cat((k_bias, v_bias), dim=0)
            self._write(ctx.with_subname("kv_proj.bias").with_quant(Quant.NONE), kv_bias)
            self._write(ctx.with_subname("q_proj.bias").with_quant(Quant.NONE), q_bias)
        else:
            qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
            self._write(ctx.with_subname("qkv_proj.bias").with_quant(Quant.NONE), qkv_bias)

        self._write(ctx.with_subname("out_proj.weight"), module.out_proj.weight)
        self._write(ctx.with_subname("out_proj.bias").with_quant(Quant.NONE), module.out_proj.bias)

    def _export_encoder_layer(self, ctx: Context, module):
        self.export_layer_norm(ctx.with_subname("norm1"), module.self_attn_layer_norm)
        self.export_layer_norm(ctx.with_subname("norm2"), module.final_layer_norm)
        self._export_attn(ctx.with_subname("attn"), module.self_attn)
        self.export_linear(ctx.with_subname("fc1"), module.fc1)
        self.export_linear(ctx.with_subname("fc2"), module.fc2)

    def _export_encoder(self, ctx: Context, module):
        self._export_conv1d(ctx.with_subname("conv1"), module.conv1)
        self._export_conv1d(ctx.with_subname("conv2"), module.conv2)
        self._export_enc_pos_embd(ctx.with_subname("pos_embd").with_quant(Quant.NONE), module.embed_positions)
        for idx, layer in enumerate(module.layers):
           self._export_encoder_layer(ctx.with_subname(f"layer{idx}"), layer)
        self.export_layer_norm(ctx.with_subname("norm"), module.layer_norm)

    def _export_decoder_layer(self, ctx: Context, module):
        self.export_layer_norm(ctx.with_subname("norm1"), module.self_attn_layer_norm)
        self.export_layer_norm(ctx.with_subname("norm2"), module.encoder_attn_layer_norm)
        self.export_layer_norm(ctx.with_subname("norm3"), module.final_layer_norm)
        self._export_attn(ctx.with_subname("self_attn"), module.self_attn)
        self._export_attn(ctx.with_subname("cross_attn"), module.encoder_attn, cross_attn=True)
        self.export_linear(ctx.with_subname("fc1"), module.fc1)
        self.export_linear(ctx.with_subname("fc2"), module.fc2)

    def _export_decoder(self, ctx: Context, module):
        self.export_embedding(ctx.with_subname("embd"), module.embed_tokens)
        self._export_enc_pos_embd(ctx.with_subname("pos_embd").with_quant(Quant.NONE), module.embed_positions)
        for idx, layer in enumerate(module.layers):
           self._export_decoder_layer(ctx.with_subname(f"layer{idx}"), layer)
        self.export_layer_norm(ctx.with_subname("norm"), module.layer_norm)

    @classmethod
    def generate_config(cls, whisper_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["whisper"] = {}
        
        section = config["whisper"]
        section["hidden_size"] = str(whisper_config.hidden_size)
        section["encoder_num_heads"] = str(whisper_config.encoder_attention_heads)
        section["encoder_ffn_dim"] = str(whisper_config.encoder_ffn_dim)
        section["encoder_num_layers"]  = str(whisper_config.encoder_layers)
        section["decoder_num_layers"]  = str(whisper_config.decoder_layers)
        section["decoder_ffn_dim"]  = str(whisper_config.decoder_ffn_dim)
        section["vocab_size"]  = str(whisper_config.vocab_size)
        section["max_tgt_length"]  = str(whisper_config.max_target_positions)

        return config

    def _export(self, ctx: Context, whisper_model):
        model = whisper_model.base_model
        self._export_encoder(ctx.with_subname("encoder"), model.encoder)
        self._export_decoder(ctx.with_subname("decoder"), model.decoder)
        self.export_linear(
            ctx.with_subname("decoder").with_subname("out_proj"),
            whisper_model.proj_out,
            has_bias=False)

    @classmethod
    def export(cls, whisper_model, fp, quant: Quant):
        config = whisper_model.config

        assert config.activation_function == "gelu"

        ctx = Context("whisper", quant=quant)
        with TensorWriter(fp) as writer:
            exporter = cls(writer)
            exporter._export(ctx, whisper_model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "whisper"
        ini_config["model"]["model_file"] = path.basename(MODEL_BIN)

        return ini_config

HELLO_URL = "https://upload.wikimedia.org/wikipedia/commons/9/9a/En-us-hello-2.ogg"

def run_whisper(huggingface_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = huggingface_name
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_file = path.join(tempfile.gettempdir(), "en-us-hello.ogg")
    if not path.exists(audio_file):
        urllib.request.urlretrieve(HELLO_URL, audio_file)

    print(pipe(audio_file))

MODEL_NAME = "openai/whisper-large-v2"
MODEL_BIN = "model.bin"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"

if __name__ == '__main__':
    from transformers import AutoTokenizer


    parser = argparse.ArgumentParser(description='export whisper model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the whisper model name in huggingface.', default=MODEL_NAME)
    parser.add_argument('-quant', type=Quant.parse, help='quantization type, "q4" or "none"', default=Quant.Q4)
    parser.add_argument('-output', type=str, help='output file name.', default="whisper.llmpkg")
    parser.add_argument('-run', action="store_true")
    args = parser.parse_args()

    if args.run:
        run_whisper(args.huggingface_name)
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_name, trust_remote_code=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.huggingface_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model = model.eval()

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        whisper_tokenizer = read_transformers_fast_bpe_model(args.huggingface_name)

        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = WhisperExporter.export(model, fp, args.quant)

        config["whisper"]["eot_token_id"] = str(tokenizer.eos_token_id)
        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            whisper_tokenizer.save(fp)
        
        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            whisper_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))
