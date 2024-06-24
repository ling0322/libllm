package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ling0322/libllm/go/chat"
	"github.com/ling0322/libllm/go/llm"
)

var gModelPath string
var gDevice string

var gLocale string

func main() {
	flag.StringVar(&gModelPath, "model", "", "path of model file (.llmpkg)")
	flag.StringVar(&gDevice, "device", "audo", "inference device (cpu|cuda|audo)")
	flag.Parse()

	localizer := NewLocalizer()

	var device llm.Device
	if strings.ToLower(gDevice) == "cpu" {
		device = llm.Cpu
	} else if strings.ToLower(gDevice) == "cuda" {
		device = llm.Cuda
	} else if strings.ToLower(gDevice) == "audo" {
		device = llm.Auto
	} else {
		log.Fatalf("unexpected device %s", gDevice)
	}

	// if model is empty, automatically choose a *.llmpkg file in working directory.
	if gModelPath == "" {
		llmpkgFiles, err := filepath.Glob("*.llmpkg")
		if err != nil {
			log.Fatal(err)
		}

		if len(llmpkgFiles) != 1 {
			fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
			flag.PrintDefaults()
		} else {
			gModelPath = llmpkgFiles[0]
		}
	}

	model, err := llm.NewModel(gModelPath, device)
	if err != nil {
		log.Fatal(err)
	}

	llmChat, err := chat.NewChat(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(localizer.Get(MsgInputQuestion))
	fmt.Println(localizer.Get(MsgInputQuestionNew))
	fmt.Println(localizer.Get(MsgInputQuestionSys))

	history := []chat.Message{}
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
			history = []chat.Message{}
			continue
		} else if strings.ToLower(question) == ":new" {
			fmt.Println(localizer.Get(MsgNewSession))
			history = []chat.Message{}
			continue
		} else if question == "" {
			continue
		}

		if len(history) == 0 && systemPrompt != "" {
			history = append(history, chat.Message{Role: "system", Content: systemPrompt})
		}

		history = append(history, chat.Message{Role: "user", Content: question})
		comp, err := llmChat.Chat(history)
		if err != nil {
			log.Fatal(err)
		}

		t0 := time.Now()
		answer := ""
		numToken := 0
		for comp.IsActive() {
			chunk, err := comp.GenerateNextChunk()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf(chunk.Text)
			answer += chunk.Text
			numToken++
		}
		history = append(history, chat.Message{Role: "assistant", Content: answer})
		fmt.Println()

		dur := time.Since(t0)
		fmt.Printf(
			localizer.Get(MsgStat),
			numToken,
			dur.Seconds(),
			dur.Seconds()*1000/float64(numToken),
		)
	}
}
