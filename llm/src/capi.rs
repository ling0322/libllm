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

//! The public C interface, which is what a program that is not written in Rust talks to.
//!
//! This is the same ABI as the C++ `llm_v2.h`: the same struct layouts, the same names, and the
//! same rules. Structs carry their own size so that a caller built against an older header keeps
//! working, failures come back as codes with the details in a thread-local slot, and the outputs
//! handed to the callback are borrowed for the length of the call and no longer.

// The names below are the ones in `llm_v2.h`, which is what a C caller writes; renaming them to
// suit Rust's conventions would mean they no longer name the same interface.
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::sync::Once;

use flint::Device;

use crate::engine::{Engine, RequestInput};
use crate::engine_config::EngineConfig;
use crate::error::Error;
use crate::llama::LlamaForGeneration;
use crate::request::{FinishReason, GenerationConfig, RequestOutput};
use crate::zip_file::ZipFile;

pub const LLM_ERROR_INVALID_ARG: i32 = 0x0100;
pub const LLM_ERROR_ABORTED: i32 = 0x0102;

/// A borrowed string. Not NUL-terminated: the length is the length.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_string_view_t {
    pub data: *const c_char,
    pub size: i64,
}

impl llm_string_view_t {
    fn empty() -> llm_string_view_t {
        llm_string_view_t {
            data: std::ptr::null(),
            size: 0,
        }
    }

    /// Borrows a Rust string for as long as it lives, which is what the callback hands out.
    fn of(value: &str) -> llm_string_view_t {
        llm_string_view_t {
            data: if value.is_empty() {
                std::ptr::null()
            } else {
                value.as_ptr() as *const c_char
            },
            size: value.len() as i64,
        }
    }

    /// # Safety
    ///
    /// `data` must point to `size` readable bytes, which the caller keeps alive for the call.
    unsafe fn to_string(self) -> Option<String> {
        if self.size < 0 || (self.data.is_null() && self.size != 0) {
            return None;
        }
        if self.size == 0 {
            return Some(String::new());
        }

        let bytes = std::slice::from_raw_parts(self.data as *const u8, self.size as usize);
        Some(String::from_utf8_lossy(bytes).into_owned())
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum llm_device_type_t {
    LLM_DEVICE_AUTO = 0,
    LLM_DEVICE_CPU = 1,
    LLM_DEVICE_CUDA = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_engine_options_t {
    pub struct_size: i64,
    pub model_path: llm_string_view_t,
    pub device: llm_device_type_t,
    pub max_num_batched_tokens: i32,
    pub kv_cache_block_size: i32,
    pub kv_cache_memory_utilization: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_generation_config_t {
    pub struct_size: i64,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub max_tokens: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_message_t {
    pub role: llm_string_view_t,
    pub content: llm_string_view_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_request_t {
    pub struct_size: i64,
    pub request_id: llm_string_view_t,
    pub input_ids: *const i64,
    pub num_input_ids: i64,
    pub messages: *const llm_message_t,
    pub num_messages: i64,
    pub generation_config: llm_generation_config_t,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum llm_finish_reason_t {
    LLM_FINISH_REASON_NONE = 0,
    LLM_FINISH_REASON_STOP = 1,
    LLM_FINISH_REASON_LENGTH = 2,
    LLM_FINISH_REASON_CANCELLED = 3,
    LLM_FINISH_REASON_ERROR = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llm_request_output_t {
    pub request_id: llm_string_view_t,
    pub token_ids: *const i64,
    pub num_token_ids: i64,
    pub text: llm_string_view_t,
    pub finished: i32,
    pub finish_reason: llm_finish_reason_t,
    pub error_message: llm_string_view_t,
}

#[repr(C)]
pub struct llm_request_outputs_t {
    pub data: *const llm_request_output_t,
    pub size: i64,
}

/// Receives a batch of outputs that the engine owns for the length of the call.
pub type llm_stream_callback_t =
    Option<unsafe extern "C" fn(outputs: *const llm_request_outputs_t, user_data: *mut c_void)>;

/// An engine handle.
pub struct llm_engine_impl_t {
    engine: Option<Engine>,
}

pub type llm_engine_t = *mut llm_engine_impl_t;

thread_local! {
    static ERROR_CODE: RefCell<i32> = const { RefCell::new(0) };
    static ERROR_MESSAGE: RefCell<CString> = RefCell::new(CString::default());
}

fn clear_error() -> i32 {
    set_error(0, "")
}

fn set_error(code: i32, message: &str) -> i32 {
    ERROR_CODE.with(|slot| *slot.borrow_mut() = code);
    ERROR_MESSAGE.with(|slot| {
        *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap_or_default();
    });
    code
}

/// Turns a crate error into a code, keeping the message for the caller to read.
fn report(error: Error) -> i32 {
    match error {
        Error::Model(message) | Error::Format(message) => {
            set_error(LLM_ERROR_INVALID_ARG, &message)
        }
        other => set_error(LLM_ERROR_ABORTED, &other.to_string()),
    }
}

/// A caller's pointer, which the stream thread hands back to them untouched.
///
/// The C contract says the caller keeps it valid until the engine is destroyed, so carrying it
/// across threads is the caller's promise rather than something this can check.
struct UserData(*mut c_void);

unsafe impl Send for UserData {}

static INIT: Once = Once::new();

/// Sets up the operator backends. Thread-safe, and does nothing after the first call.
#[no_mangle]
pub extern "C" fn llm_init() {
    INIT.call_once(flint::init);
    clear_error();
}

/// The code of the last fallible call on this thread, or zero.
#[no_mangle]
pub extern "C" fn llm_get_last_error_code() -> i32 {
    ERROR_CODE.with(|slot| *slot.borrow())
}

/// The message that goes with it. Owned by the library, valid until the next call on this thread.
#[no_mangle]
pub extern "C" fn llm_get_last_error_message() -> *const c_char {
    ERROR_MESSAGE.with(|slot| slot.borrow().as_ptr())
}

/// Fills in the default options.
///
/// # Safety
///
/// `options` must point to a writable `llm_engine_options_t`.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_options_init(options: *mut llm_engine_options_t) {
    let Some(options) = options.as_mut() else {
        return;
    };

    let defaults = EngineConfig::default();
    *options = llm_engine_options_t {
        struct_size: std::mem::size_of::<llm_engine_options_t>() as i64,
        model_path: llm_string_view_t::empty(),
        device: llm_device_type_t::LLM_DEVICE_AUTO,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
        kv_cache_block_size: defaults.kv_cache_block_size,
        kv_cache_memory_utilization: defaults.kv_cache_memory_utilization,
    };
}

/// Fills in the default generation configuration.
///
/// # Safety
///
/// `config` must point to a writable `llm_generation_config_t`.
#[no_mangle]
pub unsafe extern "C" fn llm_generation_config_init(config: *mut llm_generation_config_t) {
    let Some(config) = config.as_mut() else {
        return;
    };

    *config = llm_generation_config_t {
        struct_size: std::mem::size_of::<llm_generation_config_t>() as i64,
        top_k: 0,
        top_p: 1.0,
        temperature: 1.0,
        max_tokens: i32::MAX,
    };
}

/// Fills in an empty request. The id and the input still have to be set.
///
/// # Safety
///
/// `request` must point to a writable `llm_request_t`.
#[no_mangle]
pub unsafe extern "C" fn llm_request_init(request: *mut llm_request_t) {
    let Some(request) = request.as_mut() else {
        return;
    };

    request.struct_size = std::mem::size_of::<llm_request_t>() as i64;
    request.request_id = llm_string_view_t::empty();
    request.input_ids = std::ptr::null();
    request.num_input_ids = 0;
    request.messages = std::ptr::null();
    request.num_messages = 0;
    llm_generation_config_init(&mut request.generation_config);
}

/// Allocates an engine handle that is not loaded yet.
///
/// # Safety
///
/// `engine` must point to a writable handle holding null.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_init(engine: *mut llm_engine_t) -> i32 {
    let Some(slot) = engine.as_mut() else {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is null");
    };
    if !slot.is_null() {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is already initialized");
    }

    *slot = Box::into_raw(Box::new(llm_engine_impl_t { engine: None }));
    clear_error()
}

/// Loads a model and starts the engine. No callback happens before this returns.
///
/// # Safety
///
/// `engine` must hold a handle from [`llm_engine_init`], `options` must point to an
/// `llm_engine_options_t` whose `struct_size` it filled in, and `user_data` must stay valid until
/// the engine is destroyed.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_load(
    engine: *mut llm_engine_t,
    options: *const llm_engine_options_t,
    callback: llm_stream_callback_t,
    user_data: *mut c_void,
) -> i32 {
    let Some(handle) = engine.as_mut().and_then(|slot| slot.as_mut()) else {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is not initialized");
    };
    let Some(options) = options.as_ref() else {
        return set_error(LLM_ERROR_INVALID_ARG, "options is null");
    };
    if !has_struct_size(
        options.struct_size,
        std::mem::size_of::<llm_engine_options_t>(),
    ) {
        return set_error(LLM_ERROR_INVALID_ARG, "invalid engine options struct_size");
    }
    let Some(callback) = callback else {
        return set_error(LLM_ERROR_INVALID_ARG, "callback is null");
    };
    if handle.engine.is_some() {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is already loaded");
    }

    let Some(model_path) = options
        .model_path
        .to_string()
        .filter(|path| !path.is_empty())
    else {
        return set_error(LLM_ERROR_INVALID_ARG, "model_path is empty");
    };
    if options.max_num_batched_tokens <= 0 {
        return set_error(
            LLM_ERROR_INVALID_ARG,
            "max_num_batched_tokens must be positive",
        );
    }
    if options.kv_cache_block_size <= 0
        || options.kv_cache_block_size & (options.kv_cache_block_size - 1) != 0
    {
        return set_error(
            LLM_ERROR_INVALID_ARG,
            "kv_cache_block_size must be a power of two",
        );
    }
    if !(0.0..=1.0).contains(&options.kv_cache_memory_utilization)
        || options.kv_cache_memory_utilization <= 0.0
    {
        return set_error(
            LLM_ERROR_INVALID_ARG,
            "kv_cache_memory_utilization must be in (0, 1]",
        );
    }

    let device = match options.device {
        llm_device_type_t::LLM_DEVICE_CPU => Device::Cpu,
        llm_device_type_t::LLM_DEVICE_CUDA => Device::Cuda,
        llm_device_type_t::LLM_DEVICE_AUTO => {
            if Device::Cuda.is_available() {
                Device::Cuda
            } else {
                Device::Cpu
            }
        }
    };

    let config = EngineConfig {
        max_num_batched_tokens: options.max_num_batched_tokens,
        kv_cache_block_size: options.kv_cache_block_size,
        kv_cache_memory_utilization: options.kv_cache_memory_utilization,
    };

    let user_data = UserData(user_data);
    let built = Engine::new(
        move || {
            // Everything below runs on the engine's own thread, which is where the model has to
            // be built and where it stays.
            let package = ZipFile::open(&model_path)?;
            let model = LlamaForGeneration::from_package(device, &package)?;
            let cache = crate::kv_cache::KVCacheManager::for_model(&model, &config)?;
            Ok((model, cache))
        },
        config.max_num_batched_tokens,
        move |outputs: &[RequestOutput]| {
            // Safety: the caller promised to keep user_data valid until the engine is destroyed,
            // and the views below borrow outputs, which outlives the call.
            unsafe { emit_outputs(callback, &user_data, outputs) };
        },
    );

    match built {
        Ok(started) => {
            handle.engine = Some(started);
            clear_error()
        }
        Err(error) => report(error),
    }
}

/// Hands a batch of outputs to the callback, borrowed for the length of the call.
unsafe fn emit_outputs(
    callback: unsafe extern "C" fn(*const llm_request_outputs_t, *mut c_void),
    user_data: &UserData,
    outputs: &[RequestOutput],
) {
    let views: Vec<llm_request_output_t> = outputs
        .iter()
        .map(|output| llm_request_output_t {
            request_id: llm_string_view_t::of(&output.request_id),
            token_ids: if output.token_ids.is_empty() {
                std::ptr::null()
            } else {
                output.token_ids.as_ptr()
            },
            num_token_ids: output.token_ids.len() as i64,
            text: llm_string_view_t::of(&output.text),
            finished: i32::from(output.finished),
            finish_reason: match output.finish_reason {
                Some(FinishReason::Stop) => llm_finish_reason_t::LLM_FINISH_REASON_STOP,
                Some(FinishReason::Length) => llm_finish_reason_t::LLM_FINISH_REASON_LENGTH,
                Some(FinishReason::Cancelled) => llm_finish_reason_t::LLM_FINISH_REASON_CANCELLED,
                Some(FinishReason::Error) => llm_finish_reason_t::LLM_FINISH_REASON_ERROR,
                Some(FinishReason::None) | None => llm_finish_reason_t::LLM_FINISH_REASON_NONE,
            },
            error_message: llm_string_view_t::of(&output.error_message),
        })
        .collect();

    let batch = llm_request_outputs_t {
        data: if views.is_empty() {
            std::ptr::null()
        } else {
            views.as_ptr()
        },
        size: views.len() as i64,
    };

    callback(&batch, user_data.0);
}

/// Adds a request. Returns without waiting for it to generate.
///
/// # Safety
///
/// `engine` must hold a loaded handle, and `request` must point to an `llm_request_t` whose
/// pointers stay valid for the length of the call.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_add_request(
    engine: *mut llm_engine_t,
    request: *const llm_request_t,
) -> i32 {
    let Some(running) = loaded_engine(engine) else {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is not loaded");
    };
    let Some(request) = request.as_ref() else {
        return set_error(LLM_ERROR_INVALID_ARG, "request is null");
    };
    if !has_struct_size(request.struct_size, std::mem::size_of::<llm_request_t>())
        || !has_struct_size(
            request.generation_config.struct_size,
            std::mem::size_of::<llm_generation_config_t>(),
        )
    {
        return set_error(LLM_ERROR_INVALID_ARG, "invalid request struct_size");
    }

    let Some(request_id) = request.request_id.to_string().filter(|id| !id.is_empty()) else {
        return set_error(LLM_ERROR_INVALID_ARG, "request_id is empty");
    };

    let has_input_ids = !request.input_ids.is_null() && request.num_input_ids > 0;
    let has_messages = !request.messages.is_null() && request.num_messages > 0;
    // One form or the other: tokens to continue, or a conversation to lay out.
    if has_input_ids == has_messages {
        return set_error(
            LLM_ERROR_INVALID_ARG,
            "exactly one request input form is required",
        );
    }
    if request.num_input_ids < 0
        || request.num_messages < 0
        || request.num_input_ids > i64::from(i32::MAX)
        || request.num_messages > i64::from(i32::MAX)
    {
        return set_error(LLM_ERROR_INVALID_ARG, "request input is too large");
    }

    let generation = request.generation_config;
    if !generation.temperature.is_finite() || generation.temperature < 0.0 {
        return set_error(
            LLM_ERROR_INVALID_ARG,
            "temperature must be finite and not negative",
        );
    }
    if generation.top_k < -1 {
        return set_error(LLM_ERROR_INVALID_ARG, "top_k must be zero, -1, or positive");
    }
    if !generation.top_p.is_finite() || generation.top_p <= 0.0 || generation.top_p > 1.0 {
        return set_error(LLM_ERROR_INVALID_ARG, "top_p must be in (0, 1]");
    }
    if generation.max_tokens < 0 {
        return set_error(LLM_ERROR_INVALID_ARG, "max_tokens must not be negative");
    }

    let config = GenerationConfig {
        top_k: generation.top_k,
        top_p: generation.top_p,
        temperature: generation.temperature,
        max_tokens: generation.max_tokens,
    };

    // A conversation is laid out and encoded on the engine's thread, since that is where the
    // tokenizer lives; here only the tokens are validated.
    let input = if has_input_ids {
        let ids = std::slice::from_raw_parts(request.input_ids, request.num_input_ids as usize);
        RequestInput::Tokens(ids.to_vec())
    } else {
        let messages = std::slice::from_raw_parts(request.messages, request.num_messages as usize);
        let mut collected = Vec::with_capacity(messages.len());
        for message in messages {
            let (Some(role), Some(content)) =
                (message.role.to_string(), message.content.to_string())
            else {
                return set_error(
                    LLM_ERROR_INVALID_ARG,
                    "messages contains an invalid role or content",
                );
            };
            if role.is_empty() {
                return set_error(
                    LLM_ERROR_INVALID_ARG,
                    "messages contains an invalid role or content",
                );
            }
            collected.push(crate::prompt::Message::new(role, content));
        }
        RequestInput::Messages(collected)
    };

    match running.add_request_input(request_id, input, config) {
        Ok(()) => clear_error(),
        Err(error) => report(error),
    }
}

/// Asks for a request to stop. Unknown and finished ids do nothing.
///
/// # Safety
///
/// `engine` must hold a loaded handle.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_abort_request(
    engine: *mut llm_engine_t,
    request_id: llm_string_view_t,
) -> i32 {
    let Some(running) = loaded_engine(engine) else {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is not loaded");
    };
    let Some(request_id) = request_id.to_string().filter(|id| !id.is_empty()) else {
        return set_error(LLM_ERROR_INVALID_ARG, "request_id is empty");
    };

    match running.abort_request(request_id) {
        Ok(()) => clear_error(),
        Err(error) => report(error),
    }
}

/// Cancels what is still running, delivers the outputs that are owed, and frees the handle.
///
/// # Safety
///
/// `engine` must hold a handle from [`llm_engine_init`], and must not be called from the callback.
#[no_mangle]
pub unsafe extern "C" fn llm_engine_destroy(engine: *mut llm_engine_t) -> i32 {
    let Some(slot) = engine.as_mut() else {
        return set_error(LLM_ERROR_INVALID_ARG, "engine is null");
    };
    if slot.is_null() {
        return clear_error();
    }

    let handle = Box::from_raw(*slot);
    *slot = std::ptr::null_mut();
    drop(handle);
    clear_error()
}

/// A struct a caller filled in is usable when it is at least as large as the one it names.
fn has_struct_size(actual: i64, expected: usize) -> bool {
    actual >= 0 && actual as u64 >= expected as u64
}

unsafe fn loaded_engine<'a>(engine: *mut llm_engine_t) -> Option<&'a Engine> {
    engine.as_ref()?.as_ref()?.engine.as_ref()
}
