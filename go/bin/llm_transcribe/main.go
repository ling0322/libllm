package main

import (
	"flag"
	"log"
	"strings"

	"github.com/ling0322/libllm/go/bin"
	"github.com/ling0322/libllm/go/llm"
)

var gModelPath string
var gDevice string
var gTask string
var gInputFile string
var gLocalizer *bin.Localizer

const (
	MsgInputQuestion = iota
	MsgInputQuestionNew
	MsgInputQuestionSys
	MsgNewSession
	MsgStat
)

var gMsgZhCn = map[int]string{
	MsgInputQuestion:    "请输入问题：",
	MsgInputQuestionNew: "    输入 ':new' 重新开始一个新的对话 (清除历史).",
	MsgInputQuestionSys: "    输入 ':sys <系统指令>' 设置对话的系统指令，并重新开始一个新的对话.",
	MsgNewSession:       "===== 新的对话 =====",
	MsgStat:             "(%d个Token, 总共耗时%.2f秒, 平均每个Token耗时%.2f毫秒)\n",
}

var gMsgEnUs = map[int]string{
	MsgInputQuestion:    "Please input your question.",
	MsgInputQuestionNew: "    Type ':new' to start a new session (clean history).",
	MsgInputQuestionSys: "    Type ':sys <system_prompt>' to set the system prompt and start a new session .",
	MsgNewSession:       "===== new session =====",
	MsgStat:             "(%d tokens, time=%.2fs, %.2fms per token)\n",
}

func main() {
	flag.StringVar(&gModelPath, "model", "", "path of model file (.llmpkg)")
	flag.StringVar(&gDevice, "device", "audo", "inference device (cpu|cuda|audo)")
	flag.StringVar(&gInputFile, "input", "", "the input audio file, only used in transcribe task")
	flag.Parse()

	var err error
	gLocalizer, err = bin.NewLocalizer(map[string]map[int]string{
		"en_us": gMsgEnUs,
		"zh_cn": gMsgZhCn,
	})
	if err != nil {
		log.Fatal(err)
	}

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

	if gModelPath == "" {
		log.Fatal("argument -model is required")
	}

	model, err := llm.NewModel(gModelPath, device)
	if err != nil {
		log.Fatal(err)
	}

	if gTask == "transcribe" {
		transcribeMain(model)
	} else {
		log.Fatalf("unexpected task: %s", gTask)
	}
}
