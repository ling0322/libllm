// LLM launcher for windows.

package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	fmt.Println(`>>==================================================<<`)
	fmt.Println(`|| ____    ____  ______  ____    ____    ____    __ ||`)
	fmt.Println(`|||    |  |    ||      >|    |  |    |  |    \  /  |||`)
	fmt.Println(`|||    |_ |    ||     < |    |_ |    |_ |     \/   |||`)
	fmt.Println(`|||______||____||______>|______||______||__/\__/|__|||`)
	fmt.Println(`>>==================================================<<`)
	fmt.Println()

	llmpkgFiles, err := filepath.Glob("*.llmpkg")
	if err != nil {
		log.Fatal(err)
	}

	if len(llmpkgFiles) > 1 {
		fmt.Fprintln(os.Stderr, "请不要在当前文件夹放多于1个的.llmpkg文件")
	} else if len(llmpkgFiles) == 0 {
		fmt.Fprintln(os.Stderr, "找不到.llmpkg文件")
	} else {
		cmd := exec.Command(".\\llm.exe", "-model", llmpkgFiles[0])
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Stdin = os.Stdin

		if err := cmd.Run(); err != nil {
			os.Stderr.WriteString("错误: " + err.Error() + "\n")
		}
	}

	fmt.Println("按回车键继续...")
	fmt.Scanln()
}
