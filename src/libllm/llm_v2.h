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
#define LLM_ERROR_INSUFFICIENT_BUFFER 0x0101
#define LLM_ERROR_ABORTED 0x0102
#define LLM_ERROR_EOF 0x0103
#define LLM_ERROR_TIMEOUT 0x0104

/// Opaque inference engine. An engine owns its model, KV cache, scheduler, active requests and
/// background threads.
typedef struct llm_engine_impl_t llm_engine_t;

/// Opaque JSON value used to pass configuration and request data across the C ABI.
typedef struct llm_json_impl_t llm_json_t;

/// Receives a batch of incremental request outputs from an engine.
///
/// `outputs_json` contains a UTF-8 JSON array of request output objects. `outputs_json_size` is its
/// size in bytes, excluding any trailing null terminator. Both are borrowed and remain valid only
/// until the callback returns. `user_data` is the pointer passed to llm_engine_new(). Callbacks for
/// one engine are serialized on its stream thread. The callback may add or abort requests on the
/// same engine, but it must not call llm_engine_free().
typedef void (*llm_stream_callback_t)(
	const char *outputs_json,
	int64_t outputs_json_size,
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

// JSON

/// Allocate a JSON value initialized to null.
///
/// @return A new JSON handle owned by the caller, or NULL on allocation failure. On failure, call
/// llm_get_last_error_code() and llm_get_last_error_message() on the same thread for details.
LLMAPI llm_json_t *llm_json_new();

/// Free a JSON handle.
///
/// Passing NULL has no effect. Any pointer returned by or stored in `j` becomes invalid.
LLMAPI void llm_json_free(llm_json_t *j);

/// Parse a null-terminated UTF-8 JSON document into a new JSON value.
///
/// @param json_str Document to parse. It is not retained after this function returns.
/// @return A new JSON handle owned by the caller, or NULL if the argument is invalid, parsing
/// fails, or allocation fails. Error details are stored in the current thread's error state.
LLMAPI llm_json_t *llm_json_parse(const char *json_str);

/// Serialize a JSON value into a caller-provided buffer.
///
/// @param j JSON value to serialize.
/// @param buf Destination buffer. The result is null-terminated on success.
/// @param buf_size Destination capacity in bytes, including the null terminator.
/// @return Zero on success, LLM_ERROR_INSUFFICIENT_BUFFER if the buffer is too small, or another
/// error code for invalid arguments or serialization failure. Error details are also stored in the
/// current thread's error state.
LLMAPI int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size);

// Engine

/// Create and start an asynchronous inference engine.
///
/// The function synchronously validates `options`, loads the model, allocates its KV cache and
/// starts the scheduler and stream threads. No callback is made before this function returns.
/// `options` is copied and may be freed immediately afterwards. Recognized options include:
/// - `filename` (string, required): model package path.
/// - `device` (string, required): `cpu`, `cuda` or `auto`.
/// - `shutdown_timeout_ms` (integer, optional): time llm_engine_free() waits for accepted requests
///   to finish; a negative value waits indefinitely and zero cancels them immediately.
///
/// @param options Engine configuration.
/// @param callback Required callback that receives batches of incremental outputs.
/// @param user_data Optional caller-owned pointer passed unchanged to `callback`. It must remain
/// valid until llm_engine_free() returns.
/// @return A running engine owned by the caller, or NULL on failure. Error details are stored in
/// the current thread's error state.
LLMAPI llm_engine_t *llm_engine_new(
	llm_json_t *options,
	llm_stream_callback_t callback,
	void *user_data);

/// Gracefully shut down and free an engine.
///
/// The engine first stops accepting requests, then waits according to `shutdown_timeout_ms` for
/// all accepted requests and pending callbacks to finish. Requests that outlive the deadline are
/// cancelled and receive final cancellation outputs before the threads stop. This function blocks
/// until all engine threads have exited and must not be called from the stream callback. The
/// engine pointer is invalid after this function returns, regardless of its return value.
///
/// @param engine Engine to free. Passing NULL has no effect and succeeds.
/// @return Zero if every request finished gracefully, LLM_ERROR_TIMEOUT if unfinished requests had
/// to be cancelled, or another error code if shutdown encountered an error.
LLMAPI int32_t llm_engine_free(llm_engine_t *engine);

/// Write static and runtime information about an engine to a JSON value.
///
/// The existing content of `info` is replaced. The result may include model identity, device and
/// effective engine configuration. This function is thread-safe and does not stop generation.
///
/// @param engine Running engine.
/// @param info Caller-owned JSON handle that receives the information.
/// @return Zero on success or an error code on invalid arguments or engine failure.
LLMAPI int32_t llm_engine_get_info(llm_engine_t *engine, llm_json_t *info);

/// Submit a generation request to an engine.
///
/// This function validates and copies `request`, enqueues it, and returns without waiting for
/// generation. The JSON handle may be freed immediately afterwards. `request_id` must be non-empty
/// and unique among unfinished requests; it is copied by the engine. The request contains `messages`
/// and may contain generation parameters such as `max_tokens`, `temperature`, `top_k` and `top_p`.
/// Outputs and request-level runtime errors are delivered asynchronously through the stream
/// callback. This function is thread-safe.
///
/// @param engine Running engine accepting requests.
/// @param request_id Null-terminated request identifier.
/// @param request Request and generation configuration.
/// @return Zero once the request is accepted, or an error code if validation or enqueueing fails.
LLMAPI int32_t llm_engine_add_request(
	llm_engine_t *engine,
	const char *request_id,
	llm_json_t *request);

/// Asynchronously cancel a request.
///
/// Cancellation is idempotent: an unknown or already finished request identifier is treated as a
/// successful no-op. An active request receives one final callback output with finish reason
/// `cancelled`, after which its KV cache blocks are released. This function is thread-safe and does
/// not wait for the final callback.
///
/// @param engine Running engine.
/// @param request_id Null-terminated identifier to cancel. It is copied before this function
/// returns.
/// @return Zero when cancellation is accepted or unnecessary, or an error code for invalid
/// arguments or engine failure.
LLMAPI int32_t llm_engine_abort_request(llm_engine_t *engine, const char *request_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus