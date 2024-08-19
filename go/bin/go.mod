module github.com/ling0322/libllm/go/bin

go 1.21

replace github.com/ling0322/libllm/go/llm => ../llm

replace github.com/ling0322/libllm/go/skill => ../skill

replace github.com/ling0322/libllm/go/ffmpegplugin => ../ffmpegplugin

require (
	github.com/asticode/go-astisub v0.26.2
	github.com/ling0322/libllm/go/llm v1.0.0
	github.com/ling0322/libllm/go/skill v1.0.0
	github.com/schollz/progressbar/v3 v3.14.5
)

require (
	github.com/asticode/go-astikit v0.20.0 // indirect
	github.com/asticode/go-astits v1.8.0 // indirect
	github.com/ling0322/libllm/go/ffmpegplugin v1.0.0 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	golang.org/x/net v0.0.0-20200904194848-62affa334b73 // indirect
	golang.org/x/sys v0.22.0 // indirect
	golang.org/x/term v0.22.0 // indirect
	golang.org/x/text v0.3.2 // indirect
)
