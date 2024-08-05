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

	"github.com/ling0322/libllm/go/skill"
)

func printTranslateUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm translate [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

type translationResult struct {
	srcText        string
	tgtText        string
	numTokens      int
	processingTime time.Duration
}

func translate(
	translator skill.Translator,
	srcLang, tgtLang skill.Lang,
	text string,
	onToken func(string)) (translationResult, error) {

	text = strings.TrimSpace(text)
	if text == "" {
		return translationResult{
			srcText:        text,
			tgtText:        "",
			numTokens:      0,
			processingTime: time.Duration(0),
		}, nil
	}

	comp, err := translator.Translate(text, srcLang, tgtLang)
	if err != nil {
		return translationResult{}, err
	}

	t0 := time.Now()
	answer := ""
	numToken := 0
	for comp.Next() {
		if onToken != nil {
			onToken(comp.Text())
		}
		answer += comp.Text()
		numToken++
	}
	if err := comp.Error(); err != nil {
		return translationResult{}, err
	}

	return translationResult{
		srcText:        text,
		tgtText:        answer,
		numTokens:      numToken,
		processingTime: time.Since(t0),
	}, nil
}

func translationMain(args []string) {
	fs := flag.NewFlagSet("", flag.ExitOnError)
	fs.Usage = func() {
		printTranslateUsage(fs)
	}

	ba := newBinArgs(fs)
	ba.addModelFlag()
	ba.addDeviceFlag()
	ba.addLangFlag()
	ba.addTargetLangFlag()
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

	translator, err := skill.NewTranslator(model)
	if err != nil {
		log.Fatal(err)
	}

	srcLang := ba.getLang()
	tgtLang := ba.getTargetLang()

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

		result, err := translate(translator, srcLang, tgtLang, question, func(s string) {
			fmt.Print(s)
		})
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println()
		fmt.Printf(
			gLocalizer.Get(MsgStat),
			result.numTokens,
			result.processingTime.Seconds(),
			result.processingTime.Seconds()*1000/float64(result.numTokens),
		)
	}
}
