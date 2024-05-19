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


class AutoPtr:
    def __init__(self, p: c_void_p, deleter) -> None:
        self._ptr = p
        self._deleter = deleter
    
    def __del__(self):
        self._deleter(self._ptr)
    
    def get(self):
        return self._ptr


def _api_wrapper(f, return_checker):
    def _invoke(*args):
        return_val = f(*args)
        if return_checker(return_val) == False:
            error_message = get_last_error_message().decode("utf-8")
            raise Exception(error_message)
        return return_val 
    return _invoke

def _rc(f):
    return _api_wrapper(f, lambda r: r == 0)

def _rp(f):
    return _api_wrapper(f, lambda r: r is not None)

CFUNC_Int = CFUNCTYPE(c_int32)
CFUNC_Ptr = CFUNCTYPE(c_void_p)
CFUNC_Str = CFUNCTYPE(c_char_p)
CFUNC_Int_Int = CFUNCTYPE(c_int32, c_int32)
CFUNC_Int_Ptr = CFUNCTYPE(c_int32, c_void_p)
CFUNC_Ptr_Ptr = CFUNCTYPE(c_void_p, c_void_p)
CFUNC_Str_Ptr = CFUNCTYPE(c_char_p, c_void_p)
CFUNC_Int_PtrPtr = CFUNCTYPE(c_int32, c_void_p, c_void_p)
CFUNC_Int_PtrStr = CFUNCTYPE(c_int32, c_void_p, c_char_p)
CFUNC_Int_PtrInt = CFUNCTYPE(c_int32, c_void_p, c_int32)
CFUNC_Int_PtrFloat = CFUNCTYPE(c_int32, c_void_p, c_float)

init = _rc(CFUNC_Int_Int(("llmInit" ,_lib)))
destroy = _rc(CFUNC_Int(("llmDestroy" ,_lib)))
get_last_error_message = CFUNC_Str(("llmGetLastErrorMessage" ,_lib))

model_new = _rp(CFUNC_Ptr(("llmModel_New" ,_lib)))
model_delete = _rc(CFUNC_Int_Ptr(("llmModel_Delete" ,_lib)))
model_set_file = _rc(CFUNC_Int_PtrStr(("llmModel_SetFile" ,_lib)))
model_set_device = _rc(CFUNC_Int_PtrInt(("llmModel_SetDevice" ,_lib)))
model_load = _rc(CFUNC_Int_Ptr(("llmModel_Load" ,_lib)))
model_get_name = _rp(CFUNC_Ptr_Ptr(("llmModel_GetName" ,_lib)))

prompt_new = _rp(CFUNC_Ptr_Ptr(("llmPrompt_New" ,_lib)))
prompt_delete = _rc(CFUNC_Int_Ptr(("llmPrompt_Delete" ,_lib)))
prompt_append_text = _rc(CFUNC_Int_PtrStr(("llmPrompt_AppendText" ,_lib)))
prompt_append_control_token = _rc(CFUNC_Int_PtrStr(("llmPrompt_AppendControlToken" ,_lib)))

completion_new = _rp(CFUNC_Ptr_Ptr(("llmCompletion_New" ,_lib)))
completion_delete = _rc(CFUNC_Int_Ptr(("llmCompletion_Delete" ,_lib)))
completion_set_prompt = _rc(CFUNC_Int_PtrPtr(("llmCompletion_SetPrompt" ,_lib)))
completion_set_top_p = _rc(CFUNC_Int_PtrFloat(("llmCompletion_SetTopP" ,_lib)))
completion_set_top_k = _rc(CFUNC_Int_PtrInt(("llmCompletion_SetTopK" ,_lib)))
completion_set_temperature = _rc(CFUNC_Int_PtrFloat(("llmCompletion_SetTemperature" ,_lib)))
completion_start = _rc(CFUNC_Int_Ptr(("llmCompletion_Start" ,_lib)))
completion_is_active = CFUNC_Int_Ptr(("llmCompletion_IsActive" ,_lib))
completion_generate_next_chunk = _rc(CFUNC_Int_PtrPtr(("llmCompletion_GenerateNextChunk" ,_lib)))

chunk_new = _rp(CFUNC_Ptr(("llmChunk_New" ,_lib)))
chunk_delete = _rc(CFUNC_Int_Ptr(("llmChunk_Delete" ,_lib)))
chunk_get_text = _rp(CFUNC_Str_Ptr(("llmChunk_GetText" ,_lib)))

init(API_VERSION)
