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
	"strings"

	"github.com/ling0322/libllm/go/skill"
)

type binArgs struct {
	fs *flag.FlagSet

	models     modelList
	device     string
	inputFile  string
	outputFile string
	lang       string
	targetLang string
}

type modelList []string

func (l *modelList) String() string {
	return fmt.Sprintf("%s", *l)
}

func (l *modelList) Set(value string) error {
	*l = append(*l, value)
	return nil
}

func newBinArgs(fs *flag.FlagSet) *binArgs {
	return &binArgs{
		fs: fs,
	}
}

func (a *binArgs) addDeviceFlag() {
	a.fs.StringVar(&a.device, "device", "auto", "inference device, either cpu, cuda or auto")
}

func (a *binArgs) getDevice() string {
	device := strings.ToLower(a.device)
	if device == "cpu" || device == "cuda" || device == "auto" {
		return device
	} else {
		slog.Error(`invalid device name: must be one of "cpu", "cuda" or "auto"`)
		a.fs.Usage()
		os.Exit(1)
	}

	return device
}

func (a *binArgs) addModelFlag() {
	a.fs.Var(&a.models, "m", "the libllm model, it could be model name or model file,"+
		" model files are with suffix \".llmpkg\". "+
		"\nFor some specific tasks like transcription-translation, it could be a list of"+
		" models, seperated by \",\".")
}

func (a *binArgs) getModel() string {
	if len(a.models) == 0 {
		slog.Error("model name (-m) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	if len(a.models) != 1 {
		slog.Error("only 1 model (-m) is expected, please check if there is any unexpected comma" +
			" \",\" in model arg (-m).")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.models[0]
}

func (a *binArgs) getModels() []string {
	for _, model := range a.models {
		if model == "" {
			slog.Error("invalid model name (-m).")
			a.fs.Usage()
			os.Exit(1)
		}
	}

	return a.models
}

func (a *binArgs) getNumModels() int {
	return len(a.models)
}

func (a *binArgs) addInputFlag() {
	a.fs.StringVar(&a.inputFile, "i", "", "the input file.")
}

func (a *binArgs) getInput() string {
	if a.inputFile == "" {
		slog.Error("input file (-i) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.inputFile
}

func (a *binArgs) tryGetInput() string {
	return a.inputFile
}

func (a *binArgs) addOutputFlag() {
	a.fs.StringVar(&a.outputFile, "o", "", "the output file.")
}

func (a *binArgs) getOutput() string {
	if a.outputFile == "" {
		slog.Error("output file (-o) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.outputFile
}

func (a *binArgs) tryGetOutput() string {
	return a.outputFile
}

func (a *binArgs) addLangFlag() {
	a.fs.StringVar(&a.lang, "lang", "", "the language of input.")
}

func (a *binArgs) getLang() skill.Lang {
	if a.lang == "" {
		slog.Error("language (-lang) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	lang, err := skill.ParseLang(a.lang)
	if err != nil {
		log.Fatal(err)
	}

	return lang
}

func (a *binArgs) getRawLang() string {
	return a.lang
}

func (a *binArgs) addTargetLangFlag() {
	a.fs.StringVar(&a.targetLang, "targetlang", "", "the target language.")
}

func (a *binArgs) getTargetLang() skill.Lang {
	if a.targetLang == "" {
		return skill.UnknownLanguage
	}

	lang, err := skill.ParseLang(a.targetLang)
	if err != nil {
		slog.Error("unsupported target language (-targetlang).")
		return skill.UnknownLanguage
	}

	return lang
}
