module github.com/ling0322/libllm/go/bin

go 1.21

replace github.com/ling0322/libllm/go/llm => ../llm

replace github.com/ling0322/libllm/go/skill => ../skill

require (
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/skill v1.0.0
)

require (
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v3 v3.14.5 // indirect
	golang.org/x/sys v0.22.0 // indirect
	golang.org/x/term v0.22.0 // indirect
)
