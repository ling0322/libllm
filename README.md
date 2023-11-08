# libLLM: Efficient inference of large language models.

[![Linux](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml/badge.svg?branch=main)](https://github.com/ling0322/libllm/actions/workflows/cmake-linux.yml)

Welcome to libLLM, an open-source project tailored for efficient inference of large language models (LLMs) on everyday PCs and mobile devices. This library is dedicated to providing powerful inference capabilities to various LLMs, enabling them to seamlessly run on a variety of devices.

欢迎使用libLLM，这是一个专为在普通个人电脑和移动设备上高效推理大型语言模型（LLM）而设计的开源项目。该库专注于为各种LLM提供强大的推理能力，使其能够在各种设备中无缝运行。

## Supported models:

- ChatGLM2
- Llama2

## Key Features:

- Optimization for Everyday Devices: libLLM is optimized to run smoothly on common PCs and mobile devices, ensuring that the power of large language models is accessible to a wide range of users.
- Versatile Inference Capabilities: The library is dedicated to providing powerful inference capabilities for diverse LLMs, catering to a variety of applications and use cases.
- Native C++ Codebase: The core functionality is implemented in C++ code, allowing for direct compilation without the need for third-party dependencies.
- No External Dependencies: Enjoy hassle-free integration as libLLM operates without relying on third-party dependencies, simplifying the deployment process.

## 特点:

- 为日常设备进行优化： libLLM经过优化，可在常见的个人电脑和移动设备上平稳运行，确保大型语言模型的强大功能面向更广泛的用户。
- 多功能的推理能力： 该库致力于为各种LLM提供强大的推理能力，满足各种应用和用例的需求。
- 本机C++代码： 核心功能采用C++编写，可直接编译，无需第三方依赖，提供更好的性能。
- 无外部依赖： libLLM在运行时无需依赖第三方库（比如BLAS），简化了部署过程，使集成更加轻松。

## Build

```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```

## Run libllm command line

```c++
$ build/src/libllm/llm/llm --ini tools/chatglm2.config 
INFO 2023-10-20T08:58:55Z lymath.cc:44] lymath: Use Avx512 backend.
INFO 2023-10-20T08:58:55Z state_map.cc:58] read state map from tools/chatglm2.q4.bin
INFO 2023-10-20T08:58:58Z state_map.cc:68] reading ... 100.0%
INFO 2023-10-20T08:58:58Z state_map.cc:69] 200 tensors read.
> 你好
 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
> 
```

## API Examples

### Python

```python
import libllm

model = libllm.Model("model/chatglm2-6b-q4/chatglm2.config")
prompt = "[Round 1]\n\n问：你好\n\n答："

for chunk in libllm.Completion(model, prompt):
    print(chunk.text, end="", flush=True)

print("\nDone!")
```

## Export Huggingface models

Here is an example of exporting ChatGLM2 model from huggingface.

```bash
$ cd tools
$ python chatglm2_exporter.py
```

Then 3 files will be exported: `chatglm2.config`, `chatglm2.q4.bin` and `chatglm2.tokenizer.bin`
