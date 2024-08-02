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
	"log"
	"os"
	"strings"

	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/skill"
)

var gModelPath string
var gDevice string
var gInputFile string
var gLang string
var gTargetLang string

func addDeviceFlag(fs *flag.FlagSet) {
	fs.StringVar(&gDevice, "device", "auto", "inference device, either cpu, cuda or auto")
}

func getDeviceArg() llm.Device {
	var device llm.Device
	if strings.ToLower(gDevice) == "cpu" {
		device = llm.Cpu
	} else if strings.ToLower(gDevice) == "cuda" {
		device = llm.Cuda
	} else if strings.ToLower(gDevice) == "auto" {
		device = llm.Auto
	} else {
		log.Fatalf("unexpected device %s", gDevice)
	}

	return device
}

func addModelFlag(fs *flag.FlagSet) {
	fs.StringVar(&gModelPath, "model", "", "the libllm model file *.llmpkg")
}

func getModelArg(fs *flag.FlagSet) string {
	if gModelPath == "" {
		fs.Usage()
		os.Exit(1)
	}

	return gModelPath
}

func addInputFlag(fs *flag.FlagSet) {
	fs.StringVar(&gInputFile, "input", "", "the input file.")
}

func getInputArg(fs *flag.FlagSet) string {
	if gInputFile == "" {
		fs.Usage()
		os.Exit(1)
	}

	return gInputFile
}

func addLangFlag(fs *flag.FlagSet) {
	fs.StringVar(&gLang, "lang", "", "the language of input.")
}

func getLangArg(fs *flag.FlagSet) skill.Lang {
	if gLang == "" {
		fs.Usage()
		os.Exit(1)
	}

	lang, err := skill.ParseLang(gLang)
	if err != nil {
		log.Fatal(err)
	}

	return lang
}

func addTargetLangFlag(fs *flag.FlagSet) {
	fs.StringVar(&gTargetLang, "targetlang", "", "the target language.")
}

func getTergatLangArg(fs *flag.FlagSet) skill.Lang {
	if gTargetLang == "" {
		fs.Usage()
		os.Exit(1)
	}

	lang, err := skill.ParseLang(gTargetLang)
	if err != nil {
		log.Fatal(err)
	}

	return lang
}
