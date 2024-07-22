module github.com/ling0322/libllm/go/bin

go 1.21

toolchain go1.22.5

replace github.com/ling0322/libllm/go/llm => ../llm

replace github.com/ling0322/libllm/go/llmtasks => ../llmtasks

require (
	github.com/ling0322/libllm/go/i18n v0.0.0-20240626100001-bf6e1d13e2be
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/llmtasks v1.0.0
)
