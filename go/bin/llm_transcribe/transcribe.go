// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/llmtasks"
)

func transcribeMain(model llm.Model) {
	if gInputFile == "" {
		log.Fatal("argument -input is required for transcribe task")
	}

	transcriber := llmtasks.NewWhisper(model)
	data, err := os.ReadFile(gInputFile)
	if err != nil {
		log.Fatal(err)
	}

	comp, err := transcriber.Transcribe(data, llmtasks.TranscriptionConfig{Language: "english"})
	if err != nil {
		log.Fatal(err)
	}

	for comp.Next() {
		fmt.Print(comp.Text())
	}
	if err := comp.Error(); err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
