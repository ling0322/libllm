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
	"path/filepath"
	"runtime"
	"time"

	"github.com/ling0322/libllm/go/llm"
	"github.com/schollz/progressbar/v3"
)

var ErrInvalidModelName = errors.New("invalid model name")
var ModelCacheDir = getModelCacheDir()

var modelUrls = map[string]string{
	"index:chat:q4":       "https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/resolve/main/bilibili-index-1.9b-chat-q4.llmpkg",
	"index:character:q4":  "https://huggingface.co/ling0322/bilibili-index-1.9b-libllm/resolve/main/bilibili-index-1.9b-character-q4.llmpkg",
	"whisper:large-v3:q4": "https://huggingface.co/ling0322/whisper-libllm/resolve/main/whisper-large-v3-q4.llmpkg",
	"qwen:7b:q4":          "https://huggingface.co/ling0322/qwen-libllm/resolve/main/qwen2-7b-instruct-q4.llmpkg",
	"qwen:1.5b:q4":        "https://huggingface.co/ling0322/qwen-libllm/resolve/main/qwen2-1.5b-instruct-q4.llmpkg",
	"llama3.2:1b:q4":      "https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-1b-instruct-q4.llmpkg",
	"llama3.2:3b:q4":      "https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-3b-instruct-q4.llmpkg",
}

var modelMsUrls = map[string]string{
	"index:chat:q4":       "https://modelscope.cn/models/ling0322/bilibili-index-libllm/resolve/master/bilibili-index-1.9b-chat-q4.llmpkg",
	"index:character:q4":  "https://modelscope.cn/models/ling0322/bilibili-index-libllm/resolve/master/bilibili-index-1.9b-character-q4.llmpkg",
	"whisper:large-v3:q4": "https://modelscope.cn/models/ling0322/whisper-libllm/resolve/master/whisper-large-v3-q4.llmpkg",
	"qwen:7b:q4":          "https://modelscope.cn/models/ling0322/qwen2-libllm/resolve/master/qwen2-7b-instruct-q4.llmpkg",
	"qwen:1.5b:q4":        "https://modelscope.cn/models/ling0322/qwen2-libllm/resolve/master/qwen2-1.5b-instruct-q4.llmpkg",
	"llama3.2:1b:q4":      "https://modelscope.cn/models/ling0322/llama3.2-libllm/resolve/master/llama3.2-1b-instruct-q4.llmpkg",
	"llama3.2:3b:q4":      "https://modelscope.cn/models/ling0322/llama3.2-libllm/resolve/master/llama3.2-3b-instruct-q4.llmpkg",
}

var modelFilenames = map[string]string{
	"index:chat:q4":       "bilibili-index-1.9b-chat-q4.llmpkg",
	"index:character:q4":  "bilibili-index-1.9b-character-q4.llmpkg",
	"whisper:large-v3:q4": "whisper-large-v3-q4.llmpkg",
	"qwen:7b:q4":          "qwen2-7b-instruct-q4.llmpkg",
	"qwen:1.5b:q4":        "qwen2-1.5b-instruct-q4.llmpkg",
	"llama3.2:3b:q4":      "llama3.2-3b-instruct-q4.llmpkg",
	"llama3.2:1b:q4":      "llama3.2-1b-instruct-q4.llmpkg",
}

var defaultModelNames = map[string]string{
	"index":               "index:chat:q4",
	"index:chat":          "index:chat:q4",
	"index:chat:q4":       "index:chat:q4",
	"index:character":     "index:character:q4",
	"index:character:q4":  "index:character:q4",
	"whisper":             "whisper:large-v3:q4",
	"whisper:large-v3":    "whisper:large-v3:q4",
	"whisper:large-v3:q4": "whisper:large-v3:q4",
	"qwen":                "qwen:7b:q4",
	"qwen:7b":             "qwen:7b:q4",
	"qwen:7b:q4":          "qwen:7b:q4",
	"qwen:1.5b":           "qwen:1.5b:q4",
	"qwen:1.5b:q4":        "qwen:1.5b:q4",
	"llama3.2":            "llama3.2:3b:q4",
	"llama3.2:3b":         "llama3.2:3b:q4",
	"llama3.2:3b:q4":      "llama3.2:3b:q4",
	"llama3.2:1b":         "llama3.2:1b:q4",
	"llama3.2:1b:q4":      "llama3.2:1b:q4",
}

func resolveModelName(name string) (resolvedName string, err error) {
	resolvedName, ok := defaultModelNames[name]
	if !ok {
		return "", fmt.Errorf("unable to resolve model name \"%s\"", name)
	}

	return
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

func downloadFile(url, localPath, filename string) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	f, err := os.OpenFile(localPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	bar := progressbar.DefaultBytes(
		resp.ContentLength,
		filename,
	)
	_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	if err != nil {
		return err
	}

	return nil
}

func isInChina() bool {
	client := http.Client{
		Timeout: 5 * time.Second,
	}

	resp, err := client.Get("https://www.google.com")
	if err != nil {
		return true
	}
	defer resp.Body.Close()

	return false
}

func downloadModel(name string) (modelPath string, err error) {
	slog.Info("download model", "name", name)

	name, err = resolveModelName(name)
	if err != nil {
		return
	}

	var url string
	var ok bool
	if isInChina() {
		url, ok = modelMsUrls[name]
	} else {
		url, ok = modelUrls[name]
	}

	if !ok {
		log.Fatal("invalid model name")
	}

	filename, ok := modelFilenames[name]
	if !ok {
		log.Fatal("invalid model name")
	}

	modelPath = path.Join(ModelCacheDir, filename)
	modelDir := path.Dir(modelPath)
	err = os.MkdirAll(modelDir, os.ModePerm)
	if err != nil {
		return "", fmt.Errorf("unable to create model cache directory: %w", err)
	}

	slog.Info("download model", "url", url)
	err = downloadFile(url, modelPath+".download", filename)
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
	name, err = resolveModelName(name)
	if err != nil {
		return
	}

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

func autoDownloadModel(nameOrPath string) (filename string, err error) {
	if filepath.Ext(nameOrPath) == ".llmpkg" {
		filename = nameOrPath
	} else {
		filename, err = getOrDownloadModel(nameOrPath)
	}

	if err != nil {
		return
	}

	_, err = os.Stat(filename)
	if err != nil {
		return
	}

	return
}

func createModelAutoDownload(nameOrPath, device string) (*llm.Model, error) {
	modelPath, err := autoDownloadModel(nameOrPath)
	if err != nil {
		return nil, err
	}

	return llm.NewModel(modelPath, device)
}

func createASRModelAutoDownload(nameOrPath, device string) (*llm.ASRModel, error) {
	modelPath, err := autoDownloadModel(nameOrPath)
	if err != nil {
		return nil, err
	}

	return llm.NewASRModel(modelPath, device)
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
