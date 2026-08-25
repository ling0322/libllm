// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Tests for the C interface's contract: what it accepts, what it refuses, and what it says when
//! it refuses. Loading a model is covered by `tests/llama.rs` and by the C program in
//! `examples/`; what matters here is that a caller who gets something wrong is told, rather than
//! finding out later.

use std::ffi::{c_void, CStr};

use llm::capi::*;

fn last_error() -> String {
    unsafe { CStr::from_ptr(llm_get_last_error_message()) }
        .to_string_lossy()
        .into_owned()
}

fn view(text: &str) -> llm_string_view_t {
    llm_string_view_t {
        data: text.as_ptr() as *const std::ffi::c_char,
        size: text.len() as i64,
    }
}

#[test]
fn fills_in_the_defaults() {
    let mut options = std::mem::MaybeUninit::<llm_engine_options_t>::zeroed();
    unsafe { llm_engine_options_init(options.as_mut_ptr()) };
    let options = unsafe { options.assume_init() };

    assert_eq!(
        options.struct_size,
        std::mem::size_of::<llm_engine_options_t>() as i64
    );
    assert_eq!(options.max_num_batched_tokens, 2048);
    assert_eq!(options.kv_cache_block_size, 256);
    assert!(options.device == llm_device_type_t::LLM_DEVICE_AUTO);

    let mut request = std::mem::MaybeUninit::<llm_request_t>::zeroed();
    unsafe { llm_request_init(request.as_mut_ptr()) };
    let request = unsafe { request.assume_init() };

    // A request starts with no input at all; the caller has to give it one.
    assert!(request.input_ids.is_null());
    assert_eq!(request.num_input_ids, 0);
    assert!(request.messages.is_null());
    assert_eq!(request.generation_config.max_tokens, i32::MAX);
    assert_eq!(request.generation_config.top_p, 1.0);

    // A null pointer is ignored rather than dereferenced.
    unsafe { llm_engine_options_init(std::ptr::null_mut()) };
    unsafe { llm_request_init(std::ptr::null_mut()) };
}

#[test]
fn refuses_a_handle_that_is_not_initialized() {
    let mut engine: llm_engine_t = std::ptr::null_mut();

    assert_eq!(
        unsafe { llm_engine_init(std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("null"), "{}", last_error());

    assert_eq!(unsafe { llm_engine_init(&mut engine) }, 0);
    assert!(!engine.is_null());
    assert_eq!(llm_get_last_error_code(), 0);

    // Initializing the same handle twice would leak the first engine.
    assert_eq!(
        unsafe { llm_engine_init(&mut engine) },
        LLM_ERROR_INVALID_ARG
    );

    // Nothing can be asked of an engine that has not loaded a model.
    let mut request = std::mem::MaybeUninit::<llm_request_t>::zeroed();
    unsafe { llm_request_init(request.as_mut_ptr()) };
    let mut request = unsafe { request.assume_init() };
    request.request_id = view("r1");
    let ids = [1i64, 2];
    request.input_ids = ids.as_ptr();
    request.num_input_ids = 2;

    assert_eq!(
        unsafe { llm_engine_add_request(&mut engine, &request) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("not loaded"), "{}", last_error());

    assert_eq!(unsafe { llm_engine_destroy(&mut engine) }, 0);
    assert!(engine.is_null(), "destroying clears the handle");
    // Destroying twice is allowed, which is what lets a caller clean up unconditionally.
    assert_eq!(unsafe { llm_engine_destroy(&mut engine) }, 0);
}

#[test]
fn refuses_options_it_cannot_use() {
    let mut engine: llm_engine_t = std::ptr::null_mut();
    assert_eq!(unsafe { llm_engine_init(&mut engine) }, 0);

    unsafe extern "C" fn callback(_: *const llm_request_outputs_t, _: *mut c_void) {}

    let mut options = std::mem::MaybeUninit::<llm_engine_options_t>::zeroed();
    unsafe { llm_engine_options_init(options.as_mut_ptr()) };
    let mut options = unsafe { options.assume_init() };
    options.model_path = view("/nonexistent.llmpkg");

    // A callback is what the outputs are for, so there is no point starting without one.
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &options, None, std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("callback"), "{}", last_error());

    let mut wrong = options;
    wrong.model_path = llm_string_view_t {
        data: std::ptr::null(),
        size: 0,
    };
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &wrong, Some(callback), std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("model_path"), "{}", last_error());

    // A block size that is not a power of two cannot address a position by shifting.
    let mut wrong = options;
    wrong.kv_cache_block_size = 100;
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &wrong, Some(callback), std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("power of two"), "{}", last_error());

    let mut wrong = options;
    wrong.max_num_batched_tokens = 0;
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &wrong, Some(callback), std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );

    let mut wrong = options;
    wrong.kv_cache_memory_utilization = 1.5;
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &wrong, Some(callback), std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );

    // A struct from an older header is refused rather than read past its end.
    let mut stale = options;
    stale.struct_size = 8;
    assert_eq!(
        unsafe { llm_engine_load(&mut engine, &stale, Some(callback), std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert!(last_error().contains("struct_size"), "{}", last_error());

    // A model that is not there fails the load rather than the first request.
    assert_ne!(
        unsafe { llm_engine_load(&mut engine, &options, Some(callback), std::ptr::null_mut()) },
        0
    );

    assert_eq!(unsafe { llm_engine_destroy(&mut engine) }, 0);
}

#[test]
fn keeps_the_error_of_the_last_call_on_this_thread() {
    llm_init();
    assert_eq!(llm_get_last_error_code(), 0);

    let mut engine: llm_engine_t = std::ptr::null_mut();
    assert_eq!(
        unsafe { llm_engine_init(std::ptr::null_mut()) },
        LLM_ERROR_INVALID_ARG
    );
    assert_eq!(llm_get_last_error_code(), LLM_ERROR_INVALID_ARG);

    // A call that succeeds clears what the last failure left behind.
    assert_eq!(unsafe { llm_engine_init(&mut engine) }, 0);
    assert_eq!(llm_get_last_error_code(), 0);
    assert!(last_error().is_empty());

    assert_eq!(unsafe { llm_engine_destroy(&mut engine) }, 0);
}
