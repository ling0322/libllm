# libLLM: Efficient inference of large language models.

[![Linux](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml) [![Windows](https://github.com/ling0322/libllm/actions/workflows/cmake-windows.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-windows.yml) [![macOS](https://github.com/ling0322/libllm/actions/workflows/cmake-darwin.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-darwin.yml)

Welcome to libLLM, an open-source project designed for efficient inference of large language models (LLM) on ordinary personal computers and mobile devices. The core is implemented in C++14, without any third-party dependencies (such as BLAS or SentencePiece), enabling seamless operation across a variety of devices.

欢迎使用libLLM，这是一个专为在普通个人电脑和移动设备上高效推理大型语言模型（LLM）而设计的开源项目。核心使用C++14编写，没有第三方依赖（BLAS、SentencePiece等），能在各种设备中无缝运行。

## Model download:

| Model       | Download       |  llm Command  |
|-------------|----------------|---------------|
| Index-1.9B-Character (Role-playing) | [🤗[HF](https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/blob/main/bilibili-index-1.9b-character-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/bilibili-index-libllm/file/view/master?fileName=bilibili-index-1.9b-character-q4.llmpkg&status=2)] | llm chat -m index:character |
| Index-1.9B-Chat | [🤗[HF](https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/blob/main/bilibili-index-1.9b-chat-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/bilibili-index-libllm/file/view/master?fileName=bilibili-index-1.9b-chat-q4.llmpkg&status=2)] | llm chat -m index |
| Qwen2-1.5B-Instruct | [🤗[HF](https://huggingface.co/ling0322/qwen-libllm/blob/main/qwen2-1.5b-instruct-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/qwen2-libllm/file/view/master?fileName=qwen2-1.5b-instruct-q4.llmpkg&status=2)] | llm chat -m qwen:1.5b |
| Qwen2-7B-Instruct | [🤗[HF](https://huggingface.co/ling0322/qwen-libllm/blob/main/qwen2-7b-instruct-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/qwen2-libllm/file/view/master?fileName=qwen2-7b-instruct-q4.llmpkg&status=2)] | llm chat -m qwen:7b |
| Llama3.2-1B-Instruct | [🤗[HF](https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-1b-instruct-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/whisper-libllm/file/view/master?fileName=whisper-large-v3-q4.llmpkg&status=2)] | llm chat -m llama3.2:1b |
| Llama3.2-3B-Instruct | [🤗[HF](https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-3b-instruct-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/whisper-libllm/file/view/master?fileName=whisper-large-v3-q4.llmpkg&status=2)] | llm chat -m llama3.2 |
| Whisper-large-v3 | [🤗[HF](https://huggingface.co/ling0322/whisper-libllm/resolve/main/whisper-large-v3-q4.llmpkg)] [[MS](https://modelscope.cn/models/ling0322/whisper-libllm/file/view/master?fileName=whisper-large-v3-q4.llmpkg&status=2)] |  llm transcribe -m whisper |

`HF` = HuggingFace, `MS` = ModelScope

## Kernel support matrix

| OS       |  Platform | CUDA       |  avx2  |  avx512 | asimdhp |
|----------|-----------|------------|--------|---------|---------|
| Linux    | x64       | ✅         | ✅     | ✅       |         |
| Windows  | x64       | ✅         | ✅     | ✅       |         |
| macOS    | arm64     |            |        |         | ✅      |

## Recent updates

- [2024-09-28] Support Llama3.2 models.
- [2024-08-12] Support Whisper models.
- [2024-08-02] Support the translation command in llm.
- [2024-07-30] Support model downloading from huggingface. For example, `llm chat -model index-character` will automatically download the `index-character` model from 🤗[Huggingface](https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/blob/main/bilibili-index-1.9b-chat-q4.llmpkg).

## Quickstart

To run and chat with Bilibili-Index-1.9B-Character:

```bash
$ llm chat -m index-character
```

It will automatically download the `Bilibili-Index-1.9B-Character` from Huggingface or ModelScope (in China), and start the chat CLI in llm.

## 开始

与`Bilibili-Index-1.9B-Character`模型聊天：

```bash
$ llm chat -m index-character
```

`llm`会自动从Huggingface或者ModelScope（如果是中国IP）下载模型`Bilibili-Index-1.9B-Character`, 并且开始与它对话。

## llm command line

```bash
$ src/libllm/llm chat -m index-character
INFO 2024-07-30T12:02:28Z interface.cc:67] ISA support: AVX2=1 F16C=1 AVX512F=1
INFO 2024-07-30T12:02:28Z interface.cc:71] Use Avx512 backend.
INFO 2024-07-30T12:02:30Z matmul.cc:43] Use GEMM from cuBLAS.
INFO 2024-07-30T12:02:30Z cuda_operators.cc:51] cuda numDevices = 2
INFO 2024-07-30T12:02:30Z cuda_operators.cc:52] cuda:0 maxThreadsPerMultiProcessor = 2048
INFO 2024-07-30T12:02:30Z cuda_operators.cc:54] cuda:0 multiProcessorCount = 20
INFO 2024-07-30T12:02:30Z thread_pool.cc:73] ThreadPool started. numThreads=20
INFO 2024-07-30T12:02:30Z llm.cc:204] read model package: /home/xiaoych/.libllm/models/bilibili-index-1.9b-character-q4.llmpkg
INFO 2024-07-30T12:02:30Z model_for_generation.cc:43] model_type = index
INFO 2024-07-30T12:02:30Z model_for_generation.cc:44] device = cuda
INFO 2024-07-30T12:02:31Z state_map.cc:66] 220 tensors read.
Please input your question.
    Type ':new' to start a new session (clean history).
    Type ':sys <system_prompt>' to set the system prompt and start a new session .
> hi
您好！我是Index，请问有什么我可以帮助您的吗？
(12 tokens, time=0.76s, 63.47ms per token)
> 
```

## Build

### libLLM CPU only

```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```

#### For macOS

Please brew install OpenMP before cmake. NOTE: currently libllm macOS expected to be very slow since there is no aarch64 kernel for it.

```bash
% brew install libomp
% export OpenMP_ROOT=$(brew --prefix)/opt/libomp
% mkdir build && cd build
% cmake ..
% make -j
```

### Build with CUDA

NOTE: specify `-DCUDAToolkit_ROOT=<CUDA-DIR>` if there is multiple CUDA versions in your OS.

Recommand versions are:
- CUDA: 11.7

```bash
$ mkdir build && cd build
$ cmake -DWITH_CUDA=ON [-DCUDAToolkit_ROOT=<CUDA-DIR>] ..
$ make -j
```

## API Examples

### Python

```python
from libllm import Model, ControlToken

model = Model("tools/bilibili_index.llmpkg")
prompt = [ControlToken("<|reserved_0|>"), "hi", ControlToken("<|reserved_1|>")]

for chunk in model.complete(prompt):
    print(chunk.text, end="", flush=True)

print("\nDone!")
```

### Go

```go
package main

import (
    "fmt"
    "log"

    "github.com/ling0322/libllm/go/llm"
)

func main() {
    model, err := llm.NewModel("../../tools/bilibili_index.llmpkg", llm.Auto)
    if err != nil {
        log.Fatal(err)
    }

    prompt := llm.NewPrompt()
    prompt.AppendControlToken("<|reserved_0|>")
    prompt.AppendText("hi")
    prompt.AppendControlToken("<|reserved_1|>")
    comp, err := model.Complete(llm.NewCompletionConfig(), prompt)
    if err != nil {
        log.Fatal(err)
    }

    for comp.IsActive() {
        chunk, err := comp.GenerateNextChunk()
        if err != nil {
            log.Fatal(err)
        }

        fmt.Print(chunk.Text)
    }
    fmt.Println()
}

```

## Export Huggingface models

Here is an example of exporting Index-1.9B model from huggingface.

```bash
$ cd tools
$ python bilibili_index_exporter.py \
    -huggingface_name IndexTeam/Index-1.9B-Character \
    -quant q4  \
    -output index.llmpkg 

```

Then all required modules realted to `IndexTeam/Index-1.9B-Character`, including model, tokenizer and configs will be written to `index.llmpkg`.
