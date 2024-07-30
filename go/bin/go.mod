module github.com/ling0322/libllm/go/bin

go 1.21

replace github.com/ling0322/libllm/go/llm => ../llm

replace github.com/ling0322/libllm/go/skill => ../skill

require (
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/skill v1.0.0
)
