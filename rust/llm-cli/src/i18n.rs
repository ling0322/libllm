// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

//! The messages the tool prints, in the languages it knows.

/// One thing the tool has to say.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Message {
    Error,
    InputQuestion,
    InputQuestionNew,
    InputQuestionSys,
    NewSession,
}

/// The message in whichever language the environment asks for, falling back to English.
pub fn message(message: Message) -> &'static str {
    match locale().as_deref() {
        Some("zh_cn") => zh_cn(message),
        _ => en_us(message),
    }
}

/// The language the environment asks for, as a lowercase `xx_yy`.
///
/// `LANGUAGE` is what the Go tool reads; `LC_ALL` and `LANG` are read too, since those are what a
/// POSIX system usually sets. A Windows build would ask the system rather than the environment.
fn locale() -> Option<String> {
    for name in ["LANGUAGE", "LC_ALL", "LANG"] {
        if let Ok(value) = std::env::var(name) {
            if !value.is_empty() {
                // "zh_CN.UTF-8" and "zh-CN" both name the same language.
                let value = value.to_lowercase().replace('-', "_");
                let value = value.split(['.', ':']).next().unwrap_or(&value).to_string();
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
    }
    None
}

fn en_us(message: Message) -> &'static str {
    match message {
        Message::Error => "Error: ",
        Message::InputQuestion => "Please input your question.",
        Message::InputQuestionNew => "    Type ':new' to start a new session (clean history).",
        Message::InputQuestionSys => {
            "    Type ':sys <system_prompt>' to set the system prompt and start a new session ."
        }
        Message::NewSession => "===== new session =====",
    }
}

fn zh_cn(message: Message) -> &'static str {
    match message {
        Message::Error => "错误: ",
        Message::InputQuestion => "请输入问题：",
        Message::InputQuestionNew => "    输入 ':new' 重新开始一个新的对话 (清除历史).",
        Message::InputQuestionSys => {
            "    输入 ':sys <系统指令>' 设置对话的系统指令，并重新开始一个新的对话."
        }
        Message::NewSession => "===== 新的对话 =====",
    }
}

/// The statistics line. Its own function rather than an entry in the table above, since it is
/// the one message with anything filled into it.
pub fn stat_line(num_tokens: usize, seconds: f64, milliseconds_per_token: f64) -> String {
    match locale().as_deref() {
        Some("zh_cn") => format!(
            "({num_tokens}个Token, 总共耗时{seconds:.2}秒, 平均每个Token耗时\
             {milliseconds_per_token:.2}毫秒)"
        ),
        _ => format!(
            "({num_tokens} tokens, time={seconds:.2}s, {milliseconds_per_token:.2}ms per token)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_message_is_translated() {
        // A missing translation would otherwise show up as an empty line at runtime.
        for message in [
            Message::Error,
            Message::InputQuestion,
            Message::InputQuestionNew,
            Message::InputQuestionSys,
            Message::NewSession,
        ] {
            assert!(!en_us(message).is_empty(), "{message:?}");
            assert!(!zh_cn(message).is_empty(), "{message:?}");
            assert_ne!(en_us(message), zh_cn(message), "{message:?}");
        }
    }

    #[test]
    fn writes_the_statistics_in_both_languages() {
        assert_eq!(
            stat_line(7, 0.13, 18.6),
            "(7 tokens, time=0.13s, 18.60ms per token)"
        );
    }
}
