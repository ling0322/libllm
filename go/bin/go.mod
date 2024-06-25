module github.com/ling0322/libllm/go/bin

go 1.15

replace github.com/ling0322/libllm/go/llm => ../llm
replace github.com/ling0322/libllm/go/i18n => ../i18n
replace github.com/ling0322/libllm/go/chat => ../chat

require (
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/chat v1.0.0
	github.com/ling0322/libllm/go/i18n v1.0.0
)
