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
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ling0322/libllm/go/llm"
)

func printChatUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm chat [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

func chatMain(args []string) {
	fs := flag.NewFlagSet("", flag.ExitOnError)
	fs.Usage = func() {
		printChatUsage(fs)
	}

	ba := newBinArgs(fs)

	ba.addModelFlag()
	ba.addDeviceFlag()
	_ = fs.Parse(args)

	if fs.NArg() != 0 {
		fs.Usage()
		os.Exit(1)
	}

	modelName := ba.getModel()
	model, err := createModelAutoDownload(modelName, ba.getDevice())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(gLocalizer.Get(MsgInputQuestion))
	fmt.Println(gLocalizer.Get(MsgInputQuestionNew))
	fmt.Println(gLocalizer.Get(MsgInputQuestionSys))

	history := []llm.Message{}

	// TODO: get system prompt for different models
	systemPrompt := ""
	for {
		reader := bufio.NewReader(os.Stdin)

		fmt.Print("> ")
		question, err := reader.ReadString('\n')
		if errors.Is(err, io.EOF) {
			fmt.Println()
			break
		} else if err != nil {
			log.Fatal(err)
		}
		question = strings.TrimSpace(question)
		if len(question) > 5 && strings.ToLower(question)[0:5] == ":sys " {
			systemPrompt = strings.TrimSpace(question[5:])
			history = []llm.Message{}
			continue
		} else if strings.ToLower(question) == ":new" {
			fmt.Println(gLocalizer.Get(MsgNewSession))
			history = []llm.Message{}
			continue
		} else if question == "" {
			continue
		}

		if len(history) == 0 && systemPrompt != "" {
			history = append(history, llm.Message{Role: "system", Content: systemPrompt})
		}

		history = append(history, llm.Message{Role: "user", Content: question})
		comp, err := model.Complete(history, llm.DefaultCompletionConfig())
		if err != nil {
			log.Fatal(err)
		}

		t0 := time.Now()
		answer := ""
		numToken := 0
		for comp.Next() {
			fmt.Print(comp.Chunk().Text)
			answer += comp.Chunk().Text
			numToken++
		}
		if err := comp.Err(); err != nil {
			log.Fatal(err)
		}

		history = append(history, llm.Message{Role: "assistant", Content: answer})
		fmt.Println()

		dur := time.Since(t0)
		fmt.Printf(
			gLocalizer.Get(MsgStat),
			numToken,
			dur.Seconds(),
			dur.Seconds()*1000/float64(numToken),
		)
	}
}
