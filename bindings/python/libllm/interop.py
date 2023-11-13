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

from os import path
from ctypes import CDLL, CFUNCTYPE, c_char_p, c_void_p, c_int32, c_float

LIB_NAME = "libllm-core.so"
_lib = CDLL(path.join(path.dirname(__file__), LIB_NAME))

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

def _capi(name, restype, argtypes, return_checker=None):
    prototype = CFUNCTYPE(restype, *argtypes)
    if return_checker is None:
        return prototype((name, _lib))
    else:
        api_func = prototype((name, _lib))
        def _invoke(*args):
            rptr = api_func(*args)
            if return_checker(rptr) == False:
                error_message = llm_get_last_error_message().decode("utf-8")
                raise Exception(error_message)
            return rptr 
        return _invoke
    
def _capi_rptr(name, argtypes):
    """api that returns a pointer."""
    return _capi(name, c_void_p, argtypes, lambda r: r is not None)

def _capi_rerr(name, argtypes):
    """api that returns a error code."""
    return _capi(name, c_int32, argtypes, lambda r: r == 0)

def _capi_rstr(name, argtypes):
    """api that returns a error code."""
    return _capi(name, c_char_p, argtypes, lambda r: r is not None)

llm_init = _capi_rerr("llm_init", ())
llm_destroy = _capi("llm_destroy", None, ())
llm_get_last_error_message = _capi("llm_get_last_error_message", c_char_p, ())
    
llm_model_init = _capi_rptr("llm_model_init", (c_char_p,))
llm_model_destroy = _capi("llm_model_destroy", None, (c_void_p, ))
llm_model_get_name = _capi_rstr("llm_model_destroy", (c_void_p, ))
llm_model_complete = _capi_rptr("llm_model_complete", (c_void_p, c_void_p))

llm_compl_opt_init = _capi_rptr("llm_compl_opt_init", ())
llm_compl_opt_destroy = _capi("llm_compl_opt_destroy", None, (c_void_p, ))
llm_compl_opt_set_top_p = _capi_rerr("llm_compl_opt_set_top_p", (c_void_p, c_float))
llm_compl_opt_set_temperature = _capi_rerr("llm_compl_opt_set_temperature", (c_void_p, c_float))
llm_compl_opt_set_prompt = _capi_rerr("llm_compl_opt_set_prompt", (c_void_p, c_char_p))
llm_compl_opt_set_top_k = _capi_rerr("llm_compl_opt_set_top_k", (c_void_p, c_int32))

llm_compl_destroy = _capi("llm_compl_destroy", None, (c_void_p, ))
llm_compl_is_active = _capi("llm_compl_is_active", c_int32, (c_void_p, ))
llm_compl_next_chunk = _capi_rptr("llm_compl_next_chunk", (c_void_p, ))

llm_chunk_get_text = _capi_rstr("llm_chunk_get_text", (c_void_p, ))
llm_chunk_destroy = _capi("llm_chunk_destroy", None, (c_void_p, ))

# ensure the llm library is initialized.
llm_init()
