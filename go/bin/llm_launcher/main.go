// LLM launcher for windows.

package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ling0322/libllm/go/i18n"
)

const (
	MsgErrMoreThanOnePkg = iota
	MsgErrPkgNotExist
	MsgErrError
	MsgPressEnter
)

var gMsgZhCn = map[int]string{
	MsgErrMoreThanOnePkg: "请不要在当前文件夹放多于1个的.llmpkg文件",
	MsgErrPkgNotExist:    "找不到.llmpkg文件",
	MsgErrError:          "错误: ",
	MsgPressEnter:        "按回车键继续...",
}

var gMsgEnUs = map[int]string{
	MsgErrMoreThanOnePkg: "More than one .llmpkg in current directory. Please keep only one of them",
	MsgErrPkgNotExist:    "Unable to find .llmpkg model file in current directory",
	MsgErrError:          "Error: ",
	MsgPressEnter:        "Press [Enter] to continue ...",
}

func main() {
	fmt.Println(`>>==================================================<<`)
	fmt.Println(`|| ____    ____  ______  ____    ____    ____    __ ||`)
	fmt.Println(`|||    |  |    ||      >|    |  |    |  |    \  /  |||`)
	fmt.Println(`|||    |_ |    ||     < |    |_ |    |_ |     \/   |||`)
	fmt.Println(`|||______||____||______>|______||______||__/\__/|__|||`)
	fmt.Println(`>>==================================================<<`)
	fmt.Println()

	localizer, err := i18n.NewLocalizer(map[string]map[int]string{
		"en_us": gMsgEnUs,
		"zh_cn": gMsgZhCn,
	})
	if err != nil {
		log.Fatal(err)
	}

	llmpkgFiles, err := filepath.Glob("*.llmpkg")
	if err != nil {
		log.Fatal(err)
	}

	if len(llmpkgFiles) > 1 {
		fmt.Fprintln(os.Stderr, localizer.Get(MsgErrMoreThanOnePkg))
	} else if len(llmpkgFiles) == 0 {
		fmt.Fprintln(os.Stderr, localizer.Get(MsgErrPkgNotExist))
	} else {
		cmd := exec.Command(".\\llm.exe", "-model", llmpkgFiles[0])
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Stdin = os.Stdin

		if err := cmd.Run(); err != nil {
			os.Stderr.WriteString(localizer.Get(MsgErrError) + err.Error() + "\n")
		}
	}

	fmt.Println(localizer.Get(MsgPressEnter))
	fmt.Scanln()
}
