# libLLM: Efficient inference of large language models.

Welcome to libLLM, an open-source project designed for efficient inference of large language models (LLM) on ordinary personal computers and mobile devices. The core is implemented in C++14, without any third-party dependencies (such as BLAS or SentencePiece), enabling seamless operation across a variety of devices.

## Model download:

| Model       | Download       |  llm Command  |
|-------------|----------------|---------------|
| Llama3.2-3B-Instruct | [🤗[HF](https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-3b-instruct-fp16.llmpkg)] | llm chat -m llama3.2 |

`HF` = HuggingFace

## Kernel support matrix

| OS       |  Platform | CUDA       |  avx2  |  avx512 | asimdhp |
|----------|-----------|------------|--------|---------|---------|
| Linux    | x64       | ✅         | ✅     | ✅       |         |
| Windows  | x64       | ✅         | ✅     | ✅       |         |
| macOS    | arm64     |            |        |         | ✅      |

## Recent updates

- [2024-09-28] Support Llama3.2 models.

## Quickstart

To run and chat with Llama 3.2 3B Instruct:

```bash
$ llm chat -m llama3.2
```

It will automatically download the model from Huggingface, and start the chat CLI in llm.

## llm command line

```text
$ ./llm chat -m ../tools/llama.llmpkg
INFO 2026-08-18T06:24:41Z interface.cc:67] ISA support: AVX2=1 F16C=1 AVX512F=1
INFO 2026-08-18T06:24:41Z interface.cc:71] Use Avx512 backend.
INFO 2026-08-18T06:24:41Z matmul.cc:45] Use GEMM from cuBLAS.
INFO 2026-08-18T06:24:41Z cuda_operators.cc:86] cuda numDevices = 1
INFO 2026-08-18T06:24:41Z cuda_operators.cc:87] cuda:0 maxThreadsPerMultiProcessor = 1536
INFO 2026-08-18T06:24:41Z cuda_operators.cc:89] cuda:0 multiProcessorCount = 36
INFO 2026-08-18T06:24:41Z mp_openmp.cc:36] OMP max_threads = 32
INFO 2026-08-18T06:24:41Z model_for_generation.cc:41] model_type = llama
INFO 2026-08-18T06:24:41Z model_for_generation.cc:42] device = cuda
INFO 2026-08-18T06:24:45Z state_map.cc:66] 172 tensors read.
Please input your question.
	Type ':new' to start a new session (clean history).
	Type ':sys <system_prompt>' to set the system prompt and start a new session .
> hi
How can I assist you today?
(7 tokens, time=0.27s, 38.53ms per token)
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
