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

import codecs
from enum import IntEnum
from typing import Union, List
from . import interop
from .interop import AutoPtr, LL_TRUE

class Completion:
    """A completion task that given a llm Model and complete the input prompt."""
    def __init__(self, compl_c_ptr: AutoPtr) -> None:
        self._compl_c_ptr = compl_c_ptr

    def __iter__(self):
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()
        while interop.llm_compl_is_active(self._compl_c_ptr.get()) == LL_TRUE:
            chunk_c_ptr = AutoPtr(interop.llm_compl_next_chunk(self._compl_c_ptr.get()),
                                  interop.llm_chunk_destroy)
            chunk_token = interop.llm_chunk_get_text(chunk_c_ptr.get())
            chunk_text = utf8_decoder.decode(chunk_token)
            if chunk_text:
                yield Chunk(text=chunk_text)

        # remaining bytes in utf8_decoder
        chunk_text = utf8_decoder.decode(b"", final=True)
        if chunk_text:
            yield Chunk(text=chunk_text)

class Device(IntEnum):
    CPU =  0x0000
    CUDA = 0x0100
    AUTO = 0x1f00

class SpecialToken:
    """Represents a special token in the prompt, for example <|user|>. Similiar to control tokens in
    sentencepiece model, the special token won't be generated from normal text. Pass it with
    SpecialToken(<token-name>) is the only way to append it into input prompt."""
    def __init__(self, token_name: str) -> None:
        self._name = token_name

class Model:
    """Model in libllm."""
    def __init__(self, config_file, device: Device = Device.AUTO) -> None:
        """Create an instance of Model from the libLLM model config file. When device is
        Device.AUTO, it will automatically choose the best device to load. Otherwise load model to
        the specified device.
        Args:
            config_file (str): config of the libLLM model.
            device (Device): target computation device.
        """
        model_opt_ptr = AutoPtr(interop.llm_model_opt_init(config_file.encode("utf-8")),
                                interop.llm_model_opt_destroy)
        interop.llm_model_opt_set_device(model_opt_ptr.get(), int(device))
        self._model_c_ptr = AutoPtr(interop.llm_model_init(model_opt_ptr.get()),
                                    interop.llm_model_destroy)

    @property
    def name(self):
        model_name = interop.llm_model_get_name(self._model_instance)
        return model_name.decode("utf-8")

    def complete(self, prompt: Union[str, List[Union[str, SpecialToken]]], top_k: int = 50,
                 top_p: float = 0.8, temperature: float = 1.0) -> None:
        """Compelete the prompt with given generation settings. The prompt could be either a string
        or a list of [string|SpecialToken] (when special_token is need).
        """
        prompt_ptr = self._prepare_prompt(prompt)
        compl_opt = AutoPtr(interop.llm_compl_opt_init(), interop.llm_compl_opt_destroy)
        interop.llm_compl_opt_set_top_k(compl_opt.get(), top_k)
        interop.llm_compl_opt_set_top_p(compl_opt.get(), top_p)
        interop.llm_compl_opt_set_temperature(compl_opt.get(), temperature)
        interop.llm_compl_opt_set_prompt(compl_opt.get(), prompt_ptr.get())

        compl_c_ptr = AutoPtr(interop.llm_model_complete(self._model_c_ptr.get(), compl_opt.get()),
                              interop.llm_compl_destroy)
        return Completion(compl_c_ptr)

    def _prepare_prompt(self, prompt: Union[str, List[Union[str, SpecialToken]]]):
        prompt_ptr = AutoPtr(interop.llm_prompt_init(self._model_c_ptr.get()),
                             interop.llm_prompt_destroy)
        if isinstance(prompt, str):
            interop.llm_prompt_append_text(prompt_ptr.get(), prompt.encode("utf-8"))
        elif isinstance(prompt, list):
            for block in prompt:
                if isinstance(block, str):
                    interop.llm_prompt_append_text(prompt_ptr.get(), block.encode("utf-8"))
                elif isinstance(block, SpecialToken):
                    interop.llm_prompt_append_special_token(prompt_ptr.get(), 
                                                            block._name.encode("utf-8"))
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        return prompt_ptr

            


class Chunk:
    """a chunk of text generated by llm (in Completion)."""
    def __init__(self, text: str) -> None:
        self.text = text
