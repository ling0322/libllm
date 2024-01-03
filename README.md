# libLLM: Efficient inference of large language models.

[![Linux](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml) [![Windows](https://github.com/ling0322/libllm/actions/workflows/cmake-windows.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-windows.yml)

Welcome to libLLM, an open-source project designed for efficient inference of large language models (LLM) on ordinary personal computers and mobile devices. The core is implemented in C++14, without any third-party dependencies (such as BLAS or SentencePiece), enabling seamless operation across a variety of devices.

欢迎使用libLLM，这是一个专为在普通个人电脑和移动设备上高效推理大型语言模型（LLM）而设计的开源项目。核心使用C++14编写，没有第三方依赖（BLAS、SentencePiece等），能在各种设备中无缝运行。

## Key features:

- Optimized for everyday devices: libLLM has been optimized to run smoothly on common personal computers, ensuring the powerful capabilities of large language models are accessible to a wider range of users.
- C++ code: Written in standard C++14, it is simple and efficient.
- No external dependencies: The core functionality does not require third-party dependencies (BLAS, SentencePiece, etc.), and the necessary GEMM kernels are implemented internally (avx2, avx512).
- CUDA support: Supports accelerated inference using CUDA.

## 特点

- 为日常设备进行优化：libLLM经过优化，可在常见的个人电脑上平稳运行，确保大型语言模型的强大功能面向更广泛的用户。
- C++代码：采用标准C++14编写，简单高效。
- 无外部依赖：核心功能无需第三方依赖（BLAS、SentencePiece等），所需的GEMM内核均在内部实现(avx2、avx512)。
- 支持CUDA：支持使用CUDA加速推理。

## Supported models:

| Model            | Download |
|------------------|----------|
| Llama2           | [HuggingFace](https://huggingface.co/ling0322/llama2-7b-libllm-q4/tree/main) |
| ChatGLM3-6b      | [HuggingFace](https://huggingface.co/ling0322/chatglm3-6b-libllm-q4/tree/main) |
| ChatGLM2-6b      | [HuggingFace](https://huggingface.co/ling0322/chatglm2-6b-libllm-q4/tree/main) |

## Build

### libLLM CPU only

```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```

### Build with CUDA

NOTE: specify `-DCUDAToolkit_ROOT=<CUDA-DIR>` if there is multiple CUDA versions in your OS.

Recommand versions are:
- CUDA: 11.7

```bash
$ mkdir build && cd build
$ cmake -DCUDA_ENABLED=ON [-DCUDAToolkit_ROOT=<CUDA-DIR>] ..
$ make -j
```

## Run libllm command line

```bash
$ ./src/llm/llm -config ../model/chatglm3-6b-libllm-q4/chatglm3.config 
INFO 2023-12-19T08:56:47Z lymath.cc:42] lymath: Use Avx512 backend.
INFO 2023-12-19T08:56:48Z cuda_operators.cc:46] cuda numDevices = 1
INFO 2023-12-19T08:56:48Z cuda_operators.cc:47] cuda:0 maxThreadsPerMultiProcessor = 2048
INFO 2023-12-19T08:56:48Z cuda_operators.cc:49] cuda:0 multiProcessorCount = 20
INFO 2023-12-19T08:56:48Z llm.cc:123] OMP max_threads = 20
INFO 2023-12-19T08:56:48Z bpe_model.cc:34] read tokenizer from ../model/chatglm3-6b-libllm-q4/chatglm3.tokenizer.bin
INFO 2023-12-19T08:56:48Z model_factory.cc:35] model_type = chatglm3
INFO 2023-12-19T08:56:48Z model_factory.cc:36] device = cuda
INFO 2023-12-19T08:56:48Z state_map.cc:58] read state map from ../model/chatglm3-6b-libllm-q4/chatglm3.q4.bin
INFO 2023-12-19T08:56:51Z state_map.cc:68] reading ... 100.0%
INFO 2023-12-19T08:56:51Z state_map.cc:69] 200 tensors read.
> 你好
 
 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
(29 token, time=0.92s, 31.75ms per token)
> 
```

## API Examples

### Python

```python
from libllm import Model, ControlToken

model = Model("model/chatglm3-6b-libllm-q4/chatglm3.config")
prompt = [ControlToken("<|user|>"), "\n", "你好", ControlToken("<|assistant|>")]

for chunk in model.complete(prompt):
    print(chunk.text, end="", flush=True)

print("\nDone!")
```

## Export Huggingface models

Here is an example of exporting ChatGLM3 model from huggingface.

```bash
$ cd tools
$ python chatglm_exporter.py
```

Then 3 files will be exported: `chatglm3.config`, `chatglm3.q4.bin` and `chatglm3.tokenizer.bin`
