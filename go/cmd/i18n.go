package main

const (
	MsgInputQuestion = iota
	MsgInputQuestionNew
	MsgInputQuestionSys
	MsgNewSession
	MsgStat
)

var gMsgZhCn = map[int]string{
	MsgInputQuestion:    "请输入问题：",
	MsgInputQuestionNew: "    输入 ':new' 重新开始一个新的对话 (清楚历史).",
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

type Localizer struct {
	locale string
}

func NewLocalizer() *Localizer {
	locale, err := getLocale()
	if err != nil {
		locale = ""
	}

	return &Localizer{locale}
}

func (l *Localizer) Get(messageID int) string {
	var m map[int]string
	if l.locale == "zh_cn" {
		m = gMsgZhCn
	} else {
		m = gMsgEnUs
	}

	return m[messageID]
}
