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

	"github.com/ling0322/libllm/go/chat"
	"github.com/ling0322/libllm/go/llm"
)

var gModelPath string
var gDevice string

func main() {
	flag.StringVar(&gModelPath, "model", "", "path of model file (.llmpkg)")
	flag.StringVar(&gDevice, "device", "audo", "inference device (cpu|cuda|audo)")
	flag.Parse()

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

	model, err := llm.NewModel(gModelPath, device)
	if err != nil {
		log.Fatal(err)
	}

	llmChat, err := chat.NewChat(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Please input your question.")
	fmt.Println("    Type ':new' to start a new session (clean history).")
	fmt.Println("    Type ':sys' to input the system prompt and start a new session .")

	history := []chat.Message{}
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
		if strings.ToLower(question) == ":sys" {
			fmt.Print("SYSTEM> ")
			system, err := reader.ReadString('\n')
			if errors.Is(err, io.EOF) {
				fmt.Println()
				break
			} else if err != nil {
				log.Fatal(err)
			}
			history = []chat.Message{{Role: "system", Content: system}}
			continue
		} else if strings.ToLower(question) == ":new" {
			fmt.Println("===== new session =====")
			history = []chat.Message{}
			continue
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
			"(%d tokens, time=%.2fs, %.2fms per token)\n",
			numToken,
			dur.Seconds(),
			dur.Seconds()*1000/float64(numToken),
		)
	}
}
