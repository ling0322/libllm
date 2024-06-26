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

from libllm import Model, ControlToken


model = Model("tools/llama.llmpkg")
prompt = [
    ControlToken("<|begin_of_text|>"), 
    ControlToken("<|start_header_id|>"), "user", ControlToken("<|end_header_id|>"), 
    "\n\nhi", ControlToken("<|eot_id|>"),
    ControlToken("<|start_header_id|>"), "assistant", ControlToken("<|end_header_id|>"),
    "\n\n"]

for n_output, chunk in enumerate(model.complete(prompt)):
    print(chunk.text, end="", flush=True)
    if n_output >= 100:
        break

print("\nDone!")
