// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <stdint.h>

#if defined(_WIN32)
#ifdef LIBLLM_EXPORTS
#define LLMAPI __declspec(dllexport)
#else  // LIBLLM_EXPORTS
#define LLMAPI __declspec(dllimport)
#endif  // LIBLLM_EXPORTS
#else   // _WIN32
#ifdef LIBLLM_EXPORTS
#define LLMAPI __attribute__((visibility("default")))
#else  // LIBLLM_EXPORTS
#define LLMAPI
#endif  // LIBLLM_EXPORTS
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LLM_ERROR_INVALID_ARG 0x0100
#define LLM_ERROR_ABORTED 0x0102

/// Opaque inference engine. An engine owns its model, KV cache, scheduler, active requests and
/// background threads.
typedef struct llm_engine_impl_t *llm_engine_t;

/// Borrowed UTF-8 string. `data` need not be null-terminated and may be NULL only when `size` is
/// zero.
typedef struct llm_string_view_t {
	const char *data;
	int64_t size;
} llm_string_view_t;

/// Device on which the model executes.
typedef enum llm_device_type_t {
	LLM_DEVICE_AUTO = 0,
	LLM_DEVICE_CPU = 1,
	LLM_DEVICE_CUDA = 2,
} llm_device_type_t;

/// Top-level engine configuration. Set `struct_size` to sizeof(llm_engine_options_t).
typedef struct llm_engine_options_t {
	int64_t struct_size;
	llm_string_view_t model_path;
	llm_device_type_t device;
	int32_t max_num_batched_tokens;
	int32_t kv_cache_block_size;
	float kv_cache_memory_utilization;
} llm_engine_options_t;

/// Per-request generation configuration. Set `struct_size` to
/// sizeof(llm_generation_config_t). `top_k` zero or -1 disables top-k filtering, and temperature
/// zero selects greedy decoding.
typedef struct llm_generation_config_t {
	int64_t struct_size;
	int32_t top_k;
	float top_p;
	float temperature;
	int32_t max_tokens;
} llm_generation_config_t;

/// One chat message. Both strings are copied by llm_engine_add_request().
typedef struct llm_message_t {
	llm_string_view_t role;
	llm_string_view_t content;
} llm_message_t;

/// One generation request. Set `struct_size` to sizeof(llm_request_t). The engine copies the id,
/// input, and configuration before llm_engine_add_request() returns. Exactly one input form must
/// be provided: either `input_ids`/`num_input_ids`, or `messages`/`num_messages`.
typedef struct llm_request_t {
	int64_t struct_size;
	llm_string_view_t request_id;
	const int64_t *input_ids;
	int64_t num_input_ids;
	const llm_message_t *messages;
	int64_t num_messages;
	llm_generation_config_t generation_config;
} llm_request_t;

/// Why a request stopped producing tokens.
typedef enum llm_finish_reason_t {
	LLM_FINISH_REASON_NONE = 0,
	LLM_FINISH_REASON_STOP = 1,
	LLM_FINISH_REASON_LENGTH = 2,
	LLM_FINISH_REASON_CANCELLED = 3,
	LLM_FINISH_REASON_ERROR = 4,
} llm_finish_reason_t;

/// One request's incremental output. All pointers are borrowed from the engine and remain valid
/// only until the stream callback returns.
typedef struct llm_request_output_t {
	llm_string_view_t request_id;
	const int64_t *token_ids;
	int64_t num_token_ids;
	llm_string_view_t text;
	int32_t finished;
	llm_finish_reason_t finish_reason;
	llm_string_view_t error_message;
} llm_request_output_t;

/// A batch of incremental request outputs owned by the engine.
typedef struct llm_request_outputs_t {
	const llm_request_output_t *data;
	int64_t size;
} llm_request_outputs_t;

// Structure initialization

/// Initialize engine options to the library defaults. Passing NULL has no effect.
///
/// Defaults: device=auto, max_num_batched_tokens=2048, kv_cache_block_size=256,
/// and kv_cache_memory_utilization=0.9. The model path remains empty and must be set before
/// llm_engine_load().
LLMAPI void llm_engine_options_init(llm_engine_options_t *options);

/// Initialize generation configuration to the library defaults. Passing NULL has no effect.
///
/// Defaults: top_k=0, top_p=1, temperature=1, and max_tokens=INT32_MAX.
LLMAPI void llm_generation_config_init(llm_generation_config_t *config);

/// Initialize an empty request and its nested generation configuration. Passing NULL has no
/// effect. The request id and input ids must be set before llm_engine_add_request().
LLMAPI void llm_request_init(llm_request_t *request);

/// Receives an output batch owned by the engine. `outputs` and every pointer reachable from it
/// remain valid only until the callback returns; the callback must not free or retain them.
/// `user_data` is the pointer passed to llm_engine_load(). Callbacks for one engine are serialized
/// on its stream thread. The callback may add or abort requests on the same engine, but it must not
/// call llm_engine_destroy().
typedef void (*llm_stream_callback_t)(
	const llm_request_outputs_t *outputs,
	void *user_data);

// global state

/// Initialize process-wide libllm state and available operator backends.
///
/// This function is thread-safe and idempotent. Calling another libllm function that needs global
/// state before this function is unsupported.
LLMAPI void llm_init();

/// Return the error code produced by the last fallible libllm call on the current thread.
///
/// Successful fallible calls reset the code to zero. Errors reported asynchronously for an engine
/// request are delivered in the stream callback and do not modify the caller's thread-local state.
LLMAPI int32_t llm_get_last_error_code();

/// Return the message associated with llm_get_last_error_code() on the current thread.
///
/// The returned string is owned by libllm and remains valid until the next libllm call on the same
/// thread. The caller must not modify or free it.
LLMAPI const char *llm_get_last_error_message();

// Engine

/// Allocate an unloaded inference engine handle.
///
/// @param engine Receives the caller-owned engine handle. Must not be NULL.
/// @return Zero on success, or an error code. Error details are stored in the current thread's
/// error state.
LLMAPI int32_t llm_engine_init(llm_engine_t *engine);

/// Load and start an initialized asynchronous inference engine.
///
/// The function copies and validates `options`, loads the model, allocates its KV cache and starts
/// the scheduler and stream threads. The handle must have been initialized by llm_engine_init()
/// and must not already be loaded. No callback is made before this function returns.
///
/// @param engine Initialized engine handle.
/// @param options Engine configuration. The engine copies all referenced data.
/// @param callback Required callback that receives batches of incremental outputs.
/// @param user_data Optional caller-owned pointer passed unchanged to `callback`. It must remain
/// valid until llm_engine_destroy() returns.
/// @return Zero once the engine is running, or an error code. Error details are stored in the
/// current thread's error state.
LLMAPI int32_t llm_engine_load(
	llm_engine_t *engine,
	const llm_engine_options_t *options,
	llm_stream_callback_t callback,
	void *user_data);

/// Gracefully shut down and destroy an engine.
///
/// The engine stops accepting requests and cancels every request that has not finished, so this
/// returns promptly however large the outstanding token budgets are. Cancelled requests still
/// receive their final cancellation outputs before the threads stop. This function blocks until
/// all engine threads have exited and must not be called from the stream callback. The handle is
/// set to NULL before this function returns, regardless of its return value.
///
/// @param engine Address of the engine handle. Passing NULL or an address containing NULL has no
/// effect and succeeds.
/// @return Zero on success, or an error code if shutdown encountered an error.
LLMAPI int32_t llm_engine_destroy(llm_engine_t *engine);

/// Submit a generation request to an engine.
///
/// This function validates and copies `request`, enqueues it, then returns without waiting for
/// generation. Its request id must be non-empty and unique among unfinished requests. Outputs and
/// request-level runtime errors are delivered asynchronously through the stream callback. This
/// function is thread-safe.
///
/// @param engine Running engine accepting requests.
/// @param request Request definition and generation configuration.
/// @return Zero once the request is accepted, or an error code if validation or enqueueing fails.
LLMAPI int32_t llm_engine_add_request(
	llm_engine_t *engine,
	const llm_request_t *request);

/// Asynchronously cancel a request.
///
/// Cancellation is idempotent: an unknown or already finished request identifier is a successful
/// no-op. An active request receives one final callback output with finish reason `cancelled`, after
/// which its KV cache blocks are released. This function is thread-safe and does not wait for the
/// final callback.
///
/// @param engine Running engine.
/// @param request_id Request identifier. The engine copies it before returning.
/// @return Zero when cancellation is accepted or unnecessary, or an error code for invalid
/// arguments or engine failure.
LLMAPI int32_t llm_engine_abort_request(
	llm_engine_t *engine,
	llm_string_view_t request_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus