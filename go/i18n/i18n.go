package i18n

import (
	"errors"
	"log"
)

type Localizer struct {
	locale string
	msgMap map[int]string
}

func NewLocalizer(localeMsgMap map[string]map[int]string) (*Localizer, error) {
	locale, err := getLocale()
	if err != nil {
		locale = ""
	}

	msgMap, ok := localeMsgMap[locale]
	if ok {
		return &Localizer{locale, msgMap}, nil
	}

	msgMap, ok = localeMsgMap["en_us"]
	if !ok {
		return nil, errors.New("en_us not found in the localeMsgMap")
	}

	return &Localizer{locale, msgMap}, nil
}

func (l *Localizer) Get(messageID int) string {
	msg, ok := l.msgMap[messageID]
	if !ok {
		log.Fatal("Get localized message failed")
	}

	return msg
}
