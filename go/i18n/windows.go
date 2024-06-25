//go:build windows
// +build windows

// Original source file: https://github.com/jeandeaual/go-locale/blob/master/locale_windows.go
// LICENSE: MIT: https://github.com/jeandeaual/go-locale/blob/master/LICENSE

package i18n

import (
	"strings"
	"syscall"
	"unsafe"
)

// LocaleNameMaxLength is the maximum length of a locale name on Windows.
// See https://docs.microsoft.com/en-us/windows/win32/intl/locale-name-constants.
const LocaleNameMaxLength uint32 = 85

func getWindowsLocaleFromProc(procName string) (string, error) {
	dll, err := syscall.LoadDLL("kernel32.dll")
	if err != nil {
		return "", err
	}

	proc, err := dll.FindProc(procName)
	if err != nil {
		return "", err
	}

	buffer := make([]uint16, LocaleNameMaxLength)
	ret, _, err := proc.Call(uintptr(unsafe.Pointer(&buffer[0])), uintptr(LocaleNameMaxLength))
	if ret == 0 {
		return "", err
	}

	return syscall.UTF16ToString(buffer), nil
}

func getWindowsLocale() (string, error) {
	var (
		locale string
		err    error
	)

	for _, proc := range [...]string{"GetUserDefaultLocaleName", "GetSystemDefaultLocaleName"} {
		locale, err = getWindowsLocaleFromProc(proc)
		if err == nil {
			return locale, err
		}
	}

	return locale, err
}

// GetLocale retrieves the IETF BCP 47 language tag set on the system.
func getLocale() (string, error) {
	locale, err := getWindowsLocale()
	if err != nil {
		return "", err
	}

	locale = strings.ToLower(locale)
	locale = strings.Replace(locale, "-", "_", -1)
	return locale, err
}
