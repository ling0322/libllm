//go:build !windows
// +build !windows

package main

import (
	"fmt"
	"os"
	"strings"
)

func getLocale() (string, error) {
	locale, ok := os.LookupEnv("LANGUAGE")
	if !ok || locale == "" {
		return "", fmt.Errorf("unable to get locale")
	}

	locale = strings.ToLower(locale)
	return locale, nil
}
