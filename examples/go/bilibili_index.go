package main

// The github.com/ling0322/libllm/go/llm is a wrapper for libllm shared library. It will look for
// the libllm.{so|dylib} or llm.dll in executable directory and system shared library path. To run
// this example, please ensure libllm shared library in under LD_LIBRARY_PATH, for example:
//     $ export LD_LIBRARY_PATH=../../build/src/libllm/
//     $ go run bilibili_index.go

import (
	"fmt"
	"log"

	"github.com/ling0322/libllm/go/llm"
)

func main() {
	model, err := llm.NewModel("../../tools/bilibili_index.llmpkg", llm.Auto)
	if err != nil {
		log.Fatal(err)
	}

	prompt := llm.NewPrompt()
	prompt.AppendControlToken("<|reserved_0|>")
	prompt.AppendText("hi")
	prompt.AppendControlToken("<|reserved_1|>")
	comp, err := model.Complete(llm.NewCompletionConfig(), prompt)
	if err != nil {
		log.Fatal(err)
	}

	for comp.IsActive() {
		chunk, err := comp.GenerateNextChunk()
		if err != nil {
			log.Fatal(err)
		}

		fmt.Print(chunk.Text)
	}
	fmt.Println()
}
