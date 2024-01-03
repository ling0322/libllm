# The MIT License (MIT)
#
# Copyright (c) 2023 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from os import path
from ctypes import CDLL, Structure, CFUNCTYPE, POINTER, c_char_p, c_void_p, c_int32, c_float

if os.name == "posix":
    LIB_NAME = "libllm.so"
elif os.name == "nt":
    LIB_NAME = "llm.dll"
_lib = CDLL(path.join(path.dirname(__file__), LIB_NAME))

API_VERSION = 20240101

LL_TRUE = 1
LL_FALSE = 0

CFUNC_I = CFUNCTYPE(c_int32)
CFUNC_P = CFUNCTYPE(c_void_p)
CFUNC_S = CFUNCTYPE(c_char_p)
CFUNC_IP = CFUNCTYPE(c_int32, c_void_p)
CFUNC_PP = CFUNCTYPE(c_void_p, c_void_p)
CFUNC_SP = CFUNCTYPE(c_char_p, c_void_p)
CFUNC_IPP = CFUNCTYPE(c_int32, c_void_p, c_void_p)
CFUNC_IPS = CFUNCTYPE(c_int32, c_void_p, c_char_p)
CFUNC_IPI = CFUNCTYPE(c_int32, c_void_p, c_int32)
CFUNC_IPF = CFUNCTYPE(c_int32, c_void_p, c_float)

class AutoPtr:
    def __init__(self, p: c_void_p, deleter) -> None:
        self._ptr = p
        self._deleter = deleter
    
    def __del__(self):
        self._deleter(self._ptr)
    
    def get(self):
        return self._ptr

class Api(Structure):
    _fields_ = [
        ("init", CFUNC_I),
        ("destroy", CFUNC_I),
        ("getLastErrorMessage", CFUNC_S),

        ("createModel", CFUNC_P),
        ("destroyModel", CFUNC_IP),
        ("setModelConfig", CFUNC_IPS),
        ("setModelDevice", CFUNC_IPI),
        ("loadModel", CFUNC_IP),
        ("getModelName", CFUNC_PP),

        ("createPrompt", CFUNC_PP),
        ("destroyPrompt", CFUNC_IP),
        ("appendText", CFUNC_IPS),
        ("appendControlToken", CFUNC_IPS),

        ("createCompletion", CFUNC_PP),
        ("destroyCompletion", CFUNC_IP),
        ("setPrompt", CFUNC_IPP),
        ("setTopP", CFUNC_IPF),
        ("setTopK", CFUNC_IPI),
        ("setTemperature", CFUNC_IPF),
        ("startCompletion", CFUNC_IP),
        ("isActive", CFUNC_IP),
        ("getNextChunk", CFUNC_IPP),

        ("createChunk", CFUNC_P),
        ("destroyChunk", CFUNC_IP),
        ("getChunkText", CFUNC_SP)]

get_api = CFUNCTYPE(POINTER(Api), c_int32)(("llmGetApi", _lib))
get_api_version = CFUNCTYPE(c_int32)(("llmGetApiVersion", _lib))
_api = get_api(API_VERSION)
if _api is None:
    raise Exception(f"libllm library version mismatch: {API_VERSION} expected, but {get_api_version()} found")

def _api_wrapper(f, return_checker):
    def _invoke(*args):
        return_val = f(*args)
        if return_checker(return_val) == False:
            error_message = _api.getLastErrorMessage().decode("utf-8")
            raise Exception(error_message)
        return return_val 
    return _invoke

def _rc(f):
    return _api_wrapper(f, lambda r: r == 0)

def _rp(f):
    return _api_wrapper(f, lambda r: r is not None)

init = _rc(_api.contents.init)
destroy = _rc(_api.contents.destroy)
get_last_error_message = _api.contents.getLastErrorMessage

create_model = _rp(_api.contents.createModel)
destroy_model = _rc(_api.contents.destroyModel)
set_model_config = _rc(_api.contents.setModelConfig)
set_model_device = _rc(_api.contents.setModelDevice)
load_model = _rc(_api.contents.loadModel)
get_model_name = _rp(_api.contents.getModelName)

create_prompt = _rp(_api.contents.createPrompt)
destroy_prompt = _rc(_api.contents.destroyPrompt)
append_text = _rc(_api.contents.appendText)
append_control_token = _rc(_api.contents.appendControlToken)

create_completion = _rp(_api.contents.createCompletion)
destroy_completion = _rc(_api.contents.destroyCompletion)
set_prompt = _rc(_api.contents.setPrompt)
set_top_p = _rc(_api.contents.setTopP)
set_top_k = _rc(_api.contents.setTopK)
set_temperature = _rc(_api.contents.setTemperature)
start_completion = _rc(_api.contents.startCompletion)
is_active = _api.contents.isActive
get_next_chunk = _rc(_api.contents.getNextChunk)

create_chunk = _rp(_api.contents.createChunk)
destroy_chunk = _rc(_api.contents.destroyChunk)
get_chunk_text = _rp(_api.contents.getChunkText)

init()
