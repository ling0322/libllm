# libLLM: A lightweight vLLM-style inference engine in Rust

libLLM is a lightweight LLM inference engine with a Rust runtime and optimized C++17/CUDA kernels. It provides vLLM-style serving features including continuous batching, paged KV cache, chunked prefill, request preemption, and per-request sampling.

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

## Rust example

The Rust API can load a model package and stream generated text through an `Engine` callback:

```rust
use std::io::Write;
use std::sync::mpsc::channel;

use llm::{
	Device, Engine, EngineConfig, GenerationConfig, KVCacheManager, LlamaForGeneration,
	Message, RequestInput, RequestOutput, ZipFile,
};

fn main() -> Result<(), llm::Error> {
	let device = Device::Cuda;
	let model_path = "models/llama3.2-3b-instruct-fp16.llmpkg";
	let config = EngineConfig::default();
	let (finished_tx, finished_rx) = channel();

	let engine = Engine::new(
		move || {
			let package = ZipFile::open(model_path)?;
			let model = LlamaForGeneration::from_package(device, &package)?;
			let cache = KVCacheManager::for_model(&model, &config)?;
			Ok((model, cache))
		},
		config.max_num_batched_tokens,
		move |outputs: &[RequestOutput]| {
			for output in outputs {
				print!("{}", output.text);
				let _ = std::io::stdout().flush();
				if output.finished {
					let _ = finished_tx.send(());
				}
			}
		},
	)?;

	engine.add_request_input(
		"example",
		RequestInput::Messages(vec![Message::new("user", "Why is the sky blue?")]),
		GenerationConfig::default(),
	)?;

	let _ = finished_rx.recv();
	engine.shutdown()
}
```

After completing the build steps below, run the complete example with:

```bash
cargo run --release -p llm --example chat -- \
	models/llama3.2-3b-instruct-fp16.llmpkg \
	"Why is the sky blue?"
```

See [llm/examples/chat.rs](llm/examples/chat.rs) for the complete source.

## Build

CMake drives the whole build. Configuring picks the native Flint C++/CUDA options -- which
backends to compile, where CUDA lives, what the third_party prerequisites resolve to -- and
`cmake --build` does the rest: it builds `libflint.a`, then runs `cargo build` to link it into
`llm` and the `llm` binary.

Requirements:

- CMake 3.22 or newer
- A C++17 compiler
- Rust and Cargo
- OpenMP, unless configured with `-DWITH_OPENMP=OFF`
- The bundled libunwind build:

```bash
(cd third_party && ./install_unwind.sh)
```

### CPU build

```bash
cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

The command-line executable is written to:

```text
target/release/llm
```

### CUDA build

Install the CUDA Toolkit first. FlashAttention is enabled by default for CUDA builds and must be
built once before configuring libLLM:

```bash
./third_party/install_flash_attn.sh
```

Then configure and build:

```bash
cmake -S . -B build \
	-DWITH_CUDA=ON \
	-DCUDA_ARCH_NATIVE=ON
cmake --build build --parallel
```

`CUDA_ARCH_NATIVE=ON` builds only for GPUs installed in the current machine. Omit it when
building an artifact intended for several GPU generations. If CMake cannot find the intended
CUDA installation, add:

```text
-DCUDAToolkit_ROOT=/path/to/cuda
```

To build CUDA support without FlashAttention, configure with `-DWITH_FLASH_ATTN=OFF` instead of
running `install_flash_attn.sh`.

### macOS

Install OpenMP before configuring the CPU build:

```bash
brew install libomp
export OpenMP_ROOT="$(brew --prefix)/opt/libomp"

cmake -S . -B build -DWITH_CUDA=OFF
cmake --build build --parallel
```

### Tests

Run the native C++/CUDA test suite:

```bash
cmake --build build --target unittest --parallel
./build/unittest
```

Run the Rust tests. These read the link flags CMake already wrote out, so they work without
re-running `cmake --build` -- just be sure `build/` reflects the latest C++ if you edited a kernel:

```bash
cargo test -p llm --features cli
```

The ignored Rust CUDA integration tests can be run on a CUDA machine with:

```bash
cargo test -p llm --test tensor_cuda -- --ignored
```

Some `llm` integration tests require the model and reference packages under `models/`.

### Custom native build directory

The Rust build uses `build/` by default. Point CMake at a different build directory and it builds the
CLI the same way:

```bash
cmake -S . -B out/native -DWITH_CUDA=ON
cmake --build out/native --parallel
```

To run `cargo` directly against a build directory that isn't the default `build/`, point it there
with `LIBLLM_LIB_DIR`:

```bash
LIBLLM_LIB_DIR="$PWD/out/native" cargo build -p llm --features cli --release
```
