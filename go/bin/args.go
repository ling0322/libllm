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
	"log/slog"
	"os"
	"strings"
)

type binArgs struct {
	fs *flag.FlagSet

	models modelList
	device string
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
		" model files are with suffix \".llmpkg\".")
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
