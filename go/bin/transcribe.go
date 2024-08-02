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
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/ling0322/libllm/go/skill"
)

func printTranscribeUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm transribe [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

func transcribeMain(args []string) {
	fs := flag.NewFlagSet("", flag.ExitOnError)
	fs.Usage = func() {
		printTranscribeUsage(fs)
	}

	addDeviceFlag(fs)
	addModelFlag(fs)
	addInputFlag(fs)
	_ = fs.Parse(args)

	modelFile := getModelArg(fs)
	device := getDeviceArg()
	inputFile := getInputArg(fs)

	if fs.NArg() != 0 {
		fs.Usage()
		os.Exit(1)
	}

	model, err := createModelAutoDownload(modelFile, device)
	if err != nil {
		log.Fatal(err)
	}

	fd, err := os.Open(inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer fd.Close()

	d0 := time.Now()
	transcriber := skill.NewWhisperTranscriber(model, fd)
	for transcriber.Transcribe() {
		r := transcriber.Result()
		fmt.Println(r.String())
	}

	if err = transcriber.Err(); err != nil {
		log.Fatal(err)
	}

	processingTime := time.Since(d0)
	slog.Info(
		fmt.Sprintf("processed %s audio in %s, rtf=%.3f",
			transcriber.Offset(),
			processingTime.Round(time.Millisecond),
			processingTime.Seconds()/transcriber.Offset().Seconds()))
}
