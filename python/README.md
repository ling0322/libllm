# libllm

Python binding of libllm. Github: [https://github.com/ling0322/libllm)](https://github.com/ling0322/libllm).


### Python

```python
import libllm

model = libllm.Model("model/chatglm2-6b-libllm-q4/chatglm2.config")
prompt = "[Round 1]\n\n问：你好\n\n答："

for chunk in model.complete(prompt):
    print(chunk.text, end="", flush=True)

print("\nDone!")
```
