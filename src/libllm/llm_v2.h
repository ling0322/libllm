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

/// Opaque owned buffer containing one serialized protobuf message.
typedef struct llm_proto_impl_t llm_proto_t;

/// Receives a serialized protobuf RequestOutputs message owned by the engine. `outputs` remains
/// valid only until the callback returns; the callback must not free it or retain the pointer.
/// `user_data` is the pointer passed to llm_engine_new(). Callbacks for one engine are serialized
/// on its stream thread. The callback may add or abort requests on the same engine, but it must not
/// call llm_engine_free().
typedef void (*llm_stream_callback_t)(
	const llm_proto_t *outputs,
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

// Protobuf

/// Copy one serialized protobuf message into a new caller-owned buffer.
///
/// `data` may be NULL only when `size` is zero. The bytes are not validated against a specific
/// protobuf message type at this layer.
/// @return A caller-owned protobuf buffer, or NULL on invalid arguments or allocation failure.
LLMAPI llm_proto_t *llm_proto_new(const uint8_t *data, int64_t size);

/// Free a caller-owned protobuf buffer. Passing NULL has no effect. Engine-owned callback buffers
/// must not be passed to this function.
LLMAPI void llm_proto_free(llm_proto_t *proto);

/// Return a borrowed pointer to the serialized bytes in `proto`.
///
/// The pointer remains valid for the lifetime of `proto`. Empty messages may return NULL.
LLMAPI const uint8_t *llm_proto_data(const llm_proto_t *proto);

/// Return the number of serialized bytes in `proto`, or zero for an invalid or empty buffer.
LLMAPI int64_t llm_proto_size(const llm_proto_t *proto);

// Engine

/// Create and start an asynchronous inference engine.
///
/// The function parses `options` as a protobuf EngineOptions message, loads the model, allocates
/// its KV cache and starts the scheduler and stream threads. No callback is made before this
/// function returns. The engine copies all required values, so the protobuf buffer may be freed
/// immediately afterwards.
///
/// @param options Serialized protobuf EngineOptions message.
/// @param callback Required callback that receives batches of incremental outputs.
/// @param user_data Optional caller-owned pointer passed unchanged to `callback`. It must remain
/// valid until llm_engine_free() returns.
/// @return A running engine owned by the caller, or NULL on failure. Error details are stored in
/// the current thread's error state.
LLMAPI llm_engine_t *llm_engine_new(
	const llm_proto_t *options,
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

/// Submit a generation request to an engine.
///
/// This function parses `request` as a protobuf Request message, validates and enqueues it, then
/// returns without waiting for generation. `request_id` must be non-empty and unique among
/// unfinished requests. The protobuf buffer may be freed immediately afterwards. Outputs and
/// request-level runtime errors are delivered asynchronously through the stream callback. This
/// function is thread-safe.
///
/// @param engine Running engine accepting requests.
/// @param request_id Null-terminated request identifier.
/// @param request Serialized protobuf Request message.
/// @return Zero once the request is accepted, or an error code if validation or enqueueing fails.
LLMAPI int32_t llm_engine_add_request(
	llm_engine_t *engine,
	const char *request_id,
	const llm_proto_t *request);

/// Asynchronously cancel a request.
///
/// Cancellation is idempotent: an unknown or already finished request identifier is a successful
/// no-op. An active request receives one final callback output with finish reason `cancelled`, after
/// which its KV cache blocks are released. This function is thread-safe and does not wait for the
/// final callback.
///
/// @param engine Running engine.
/// @param request_id Null-terminated request identifier.
/// @return Zero when cancellation is accepted or unnecessary, or an error code for invalid
/// arguments or engine failure.
LLMAPI int32_t llm_engine_abort_request(llm_engine_t *engine, const char *request_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus