module github.com/ling0322/libllm/go/bin

go 1.15

replace github.com/ling0322/libllm/go/llm => ../llm

replace github.com/ling0322/libllm/go/llmtasks => ../llmtasks

require (
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/llmtasks v1.0.0
)
