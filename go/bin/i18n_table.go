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

var gLocalizer = NewLocalizer(map[string]map[int]string{
	"en_us": gMsgEnUs,
	"zh_cn": gMsgZhCn,
})

const (
	MsgErrMoreThanOnePkg = iota
	MsgErrPkgNotExist
	MsgErrError
	MsgPressEnter
	MsgInputQuestion
	MsgInputQuestionNew
	MsgInputQuestionSys
	MsgNewSession
	MsgStat
)

var gMsgZhCn = map[int]string{
	MsgErrMoreThanOnePkg: "请不要在当前文件夹放多于1个的.llmpkg文件",
	MsgErrPkgNotExist:    "找不到.llmpkg文件",
	MsgErrError:          "错误: ",
	MsgPressEnter:        "按回车键继续...",
	MsgInputQuestion:     "请输入问题：",
	MsgInputQuestionNew:  "    输入 ':new' 重新开始一个新的对话 (清除历史).",
	MsgInputQuestionSys:  "    输入 ':sys <系统指令>' 设置对话的系统指令，并重新开始一个新的对话.",
	MsgNewSession:        "===== 新的对话 =====",
	MsgStat:              "(%d个Token, 总共耗时%.2f秒, 平均每个Token耗时%.2f毫秒)\n",
}

var gMsgEnUs = map[int]string{
	MsgErrMoreThanOnePkg: "More than one .llmpkg in current directory. Please keep only one of them",
	MsgErrPkgNotExist:    "Unable to find .llmpkg model file in current directory",
	MsgErrError:          "Error: ",
	MsgPressEnter:        "Press [Enter] to continue ...",
	MsgInputQuestion:     "Please input your question.",
	MsgInputQuestionNew:  "    Type ':new' to start a new session (clean history).",
	MsgInputQuestionSys:  "    Type ':sys <system_prompt>' to set the system prompt and start a new session .",
	MsgNewSession:        "===== new session =====",
	MsgStat:              "(%d tokens, time=%.2fs, %.2fms per token)\n",
}
