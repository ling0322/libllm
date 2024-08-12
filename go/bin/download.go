package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"path"
	"runtime"

	"github.com/ling0322/libllm/go/llm"
	"github.com/schollz/progressbar/v3"
)

var ErrInvalidModelName = errors.New("invalid model name")
var ModelCacheDir = getModelCacheDir()

var modelUrls = map[string]string{
	"index-chat":      "https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/resolve/main/bilibili-index-1.9b-chat-q4.llmpkg",
	"index-character": "https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/resolve/main/bilibili-index-1.9b-character-q4.llmpkg",
}

var modelFilenames = map[string]string{
	"index-chat":      "bilibili-index-1.9b-chat-q4.llmpkg",
	"index-character": "bilibili-index-1.9b-character-q4.llmpkg",
}

func getModelCacheDir() string {
	var cacheDir string
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		userDir, err := os.UserHomeDir()
		if err != nil {
			log.Fatal(err)
		}
		cacheDir = path.Join(userDir, ".libllm", "models")
	} else if runtime.GOOS == "windows" {
		binFile, err := os.Executable()
		if err != nil {
			log.Fatal(err)
		}

		binDir := path.Dir(binFile)
		cacheDir = path.Join(binDir, "models")
	}

	return cacheDir
}

func downloadModel(name string) (modelPath string, err error) {
	url, ok := modelUrls[name]
	if !ok {
		log.Fatal("invalid model name")
	}

	filename, ok := modelFilenames[name]
	if !ok {
		log.Fatal("invalid model name")
	}
	modelPath = path.Join(ModelCacheDir, filename)
	slog.Info("download model", "url", url)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	modelDir := path.Dir(modelPath)
	err = os.MkdirAll(modelDir, os.ModePerm)
	if err != nil {
		return "", fmt.Errorf("unable to create model cache directory: %w", err)
	}

	f, err := os.OpenFile(modelPath+".download", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	bar := progressbar.DefaultBytes(
		resp.ContentLength,
		"Downloading",
	)
	_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	if err != nil {
		return
	}

	err = os.Rename(modelPath+".download", modelPath)
	if err != nil {
		return
	}

	slog.Info("Save model", "path", modelPath)
	return modelPath, nil
}

// check if model exists in the cache directory. If exists, retuen the model path, otherwise,
// return the error.
func checkModelInCache(name string) (modelPath string, err error) {
	filename, ok := modelFilenames[name]
	if !ok {
		return "", ErrInvalidModelName
	}
	modelPath = path.Join(ModelCacheDir, filename)

	_, err = os.Stat(modelPath)
	if err != nil {
		return "", err
	}

	return
}

func getOrDownloadModel(name string) (modelPath string, err error) {
	modelPath, err = checkModelInCache(name)
	if err == nil {
		return
	}

	return downloadModel(name)
}

func createModelAutoDownload(nameOrPath string, device llm.Device) (llm.Model, error) {
	var modelPath string
	var err error

	_, ok := modelFilenames[nameOrPath]
	if ok {
		modelPath, err = getOrDownloadModel(nameOrPath)
	} else {
		modelPath = nameOrPath
	}

	if err != nil {
		return nil, err
	}

	_, err = os.Stat(modelPath)
	if err != nil {
		return nil, fmt.Errorf("model not exist: %s", modelPath)
	}

	return llm.NewModel(modelPath, device)
}

func printDownloadUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm download [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

func downloadMain(args []string) {
	fs := flag.NewFlagSet("", flag.ExitOnError)
	fs.Usage = func() {
		printDownloadUsage(fs)
	}

	ba := newBinArgs(fs)
	ba.addModelFlag()
	_ = fs.Parse(args)

	if fs.NArg() != 0 {
		fs.Usage()
		os.Exit(1)
	}

	modelName := ba.getModel()
	if modelPath, err := checkModelInCache(modelName); err == nil {
		fmt.Printf("model \"%s\" already downloaded. Path is \"%s\"\n", modelName, modelPath)
	}

	_, err := downloadModel(modelName)
	if err != nil {
		log.Fatal(err)
	}
}
