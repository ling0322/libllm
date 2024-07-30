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
	"os"
)

func printCommandUsage() {
	fmt.Fprintln(os.Stderr, "Usage: llm COMMAND")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "    chat           Chat with LLM")
	fmt.Fprintln(os.Stderr, "    transcribe     Transcribe audio or video file to text")
	fmt.Fprintln(os.Stderr, "    download       Download model to local")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Run 'llm COMMAND -h' for more information on a command.")
}

func main() {
	if len(os.Args) == 1 {
		printCommandUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	switch command {
	case "chat":
		chatMain(os.Args[2:])
	case "transcribe":
		transcribeMain(os.Args[2:])
	case "download":
		downloadMain(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Invalid command \"%s\"\n\n", command)
		printCommandUsage()
		os.Exit(1)
	}
}
