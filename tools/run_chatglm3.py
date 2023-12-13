from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float().cpu()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。"}])
print(response)
