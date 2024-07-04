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

    def _export_encoder(self, ctx: Context, module):
        self._export_conv1d(ctx.with_subname("conv1"), module.conv1)
        self._export_conv1d(ctx.with_subname("conv2"), module.conv2)

    @classmethod
    def generate_config(cls, whisper_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["whisper"] = {}
        
        section = config["whisper"]
        section["hidden_size"] = str(whisper_config.hidden_size)

        return config

    def _export(self, ctx: Context, module):
        self._export_encoder(ctx.with_subname("encoder"), module.encoder)

    @classmethod
    def export(cls, whisper_model, fp, quant: Quant):
        model = whisper_model.base_model
        config = whisper_model.config

        ctx = Context("whisper", quant=quant)
        with TensorWriter(fp) as writer:
            exporter = cls(writer)
            exporter._export(ctx, model)

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

MODEL_NAME = "openai/whisper-large-v3"
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