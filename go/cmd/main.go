package main

import (
	"fmt"
	"log"

	"github.com/ling0322/libllm/go/llm"
)

func main() {
	model, err := llm.NewModel("../../tools/llama.llmpkg", llm.Cuda)
	if err != nil {
		log.Fatal(err)
	}

	prompt := llm.NewPrompt()
	prompt.AppendControlToken("<|begin_of_text|>")
	prompt.AppendControlToken("<|start_header_id|>")
	prompt.AppendText("user")
	prompt.AppendControlToken("<|end_header_id|>")
	prompt.AppendText("\n\nHi")
	prompt.AppendControlToken("<|eot_id|>")
	prompt.AppendControlToken("<|start_header_id|>")
	prompt.AppendText("assistant")
	prompt.AppendControlToken("<|end_header_id|>")
	prompt.AppendText("\n\n")

	log.Println(model)

	comp, err := model.Complete(llm.NewCompletionConfig(), prompt)
	if err != nil {
		log.Fatal(err)
	}

	for comp.IsActive() {
		chunk, err := comp.GenerateNextChunk()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf(chunk.Text)
	}
}
