// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "libllm/dialog_manager.h"

#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"
#include "libllm/lut/time.h"

namespace libllm {

ChatOutput::ChatOutput()
    : numAnswerTokens(0),
      promptDuration(0.0),
      answerDuration(0.0) {
}

// -----------------------------------------------------------------------------------------------+
// class ChatGLM2PromptBuilder                                                                    |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<llm::Prompt> ChatGLM2PromptBuilder::buildPrompt(
    std::shared_ptr<llm::Model> model,
    lut::Span<const QA> history,
    const std::string &question) {
  std::string prompt;

  int round = 1;
  for (const QA &qa : history) {
    prompt += lut::sprintf("[Round %d]\n\n问：%s\n\n答：%s\n\n", round, qa.question, qa.answer);
    ++round;
  }
  prompt += lut::sprintf("[Round %d]\n\n问：%s\n\n答：", round, question);

  std::shared_ptr<llm::Prompt> pPrompt = llm::Prompt::fromModel(model);
  pPrompt->appendText(prompt);
  return pPrompt;
}

std::string ChatGLM2PromptBuilder::getStopSeq() {
  return "";
}

// -----------------------------------------------------------------------------------------------+
// class ChatGLM3PromptBuilder                                                                    |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<llm::Prompt> ChatGLM3PromptBuilder::buildPrompt(
    std::shared_ptr<llm::Model> model,
    lut::Span<const QA> history,
    const std::string &question) {
  std::shared_ptr<llm::Prompt> prompt = llm::Prompt::fromModel(model);
  for (const QA &qa : history) {
    prompt->appendControlToken("<|user|>");
    prompt->appendText("\n");
    prompt->appendText(lut::sprintf("%s", qa.question));
    prompt->appendControlToken("<|assistant|>");
    prompt->appendText("\n");
    prompt->appendText(lut::sprintf("%s", qa.answer));
  }

  prompt->appendControlToken("<|user|>");
  prompt->appendText("\n");
  prompt->appendText(lut::sprintf("%s", question));
  prompt->appendControlToken("<|assistant|>");

  return prompt;
}

std::string ChatGLM3PromptBuilder::getStopSeq() {
  return "";
}

// -----------------------------------------------------------------------------------------------+
// class LlamaPromptBuilder                                                                       |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<llm::Prompt> LlamaPromptBuilder::buildPrompt(
    std::shared_ptr<llm::Model> model,
    lut::Span<const QA> history,
    const std::string &question) {
  std::shared_ptr<llm::Prompt> prompt = llm::Prompt::fromModel(model);
  bool systemPromptAdded = false;
  for (const QA &qa : history) {
    prompt->appendControlToken("<s>");
    if (!systemPromptAdded) {
      prompt->appendText("[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n");
      systemPromptAdded = true;
    } else {
      prompt->appendText("[INST]");
    }
    prompt->appendText(qa.question + " [/INST]" + qa.answer);
    prompt->appendControlToken("</s>");
  }
  prompt->appendControlToken("<s>");
  if (!systemPromptAdded) {
    prompt->appendText("[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n");
  } else {
    prompt->appendText("[INST]");
  }
  prompt->appendText(question + " [/INST]");
  return prompt;
}

std::string LlamaPromptBuilder::getStopSeq() {
  return "";
}

// -----------------------------------------------------------------------------------------------+
// class QwenPromptBuilder                                                                        |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<llm::Prompt> QwenPromptBuilder::buildPrompt(
    std::shared_ptr<llm::Model> model,
    lut::Span<const QA> history,
    const std::string &question) {
  std::shared_ptr<llm::Prompt> prompt = llm::Prompt::fromModel(model);
  prompt->appendControlToken("<|im_start|>");
  prompt->appendText("system\nYou are a helpful assistant.");
  prompt->appendControlToken("<|im_end|>");
  for (const QA &qa : history) {
    prompt->appendText("\n");
    prompt->appendControlToken("<|im_start|>");
    prompt->appendText(lut::sprintf("user\n%s", qa.question));
    prompt->appendControlToken("<|im_end|>");
    prompt->appendText("\n");
    prompt->appendControlToken("<|im_start|>");
    prompt->appendText(lut::sprintf("assistant\n%s", qa.answer));
    prompt->appendControlToken("<|im_end|>");
  }
  prompt->appendText("\n");
  prompt->appendControlToken("<|im_start|>");
  prompt->appendText(lut::sprintf("user\n%s", question));
  prompt->appendControlToken("<|im_end|>");
  prompt->appendText("\n");
  prompt->appendControlToken("<|im_start|>");
  prompt->appendText("assistant\n");

  return prompt;
}

std::string QwenPromptBuilder::getStopSeq() {
  return "";
}

// -----------------------------------------------------------------------------------------------+
// class PromptBuilder                                                                            |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<PromptBulder> PromptBulder::create(const std::string &modelName) {
  if (modelName == "llama") return std::make_shared<LlamaPromptBuilder>();
  if (modelName == "qwen") return std::make_shared<QwenPromptBuilder>();
  if (modelName == "chatglm2") return std::make_shared<ChatGLM2PromptBuilder>();
  if (modelName == "chatglm3") return std::make_shared<ChatGLM3PromptBuilder>();

  THROW(Aborted, "unexpected model name: " + modelName);
  return nullptr;
}

// -----------------------------------------------------------------------------------------------+
// class DialogManager                                                                            |
// -----------------------------------------------------------------------------------------------+

DialogManager::DialogManager(
    std::shared_ptr<llm::Model> model,
    std::shared_ptr<PromptBulder> promptBuilder)
    : _model(model),
      _promptBuilder(promptBuilder) {
}

ChatOutput DialogManager::chat(
    const std::string &question,
    std::function<void(const std::string &)> onTokenCallback) {
  ChatOutput output;

  std::shared_ptr<llm::Prompt> prompt = _promptBuilder->buildPrompt(_model, _history, question);
  llm::CompletionConfig config;
  config.setTopK(1);

  double t0 = lut::now();
  std::shared_ptr<llm::Completion> comp = _model->complete(prompt);
  output.promptDuration = lut::now() - t0;

  std::string answer;
  std::string stopSeq = _promptBuilder->getStopSeq();
  t0 = lut::now();
  int numToken = 0;
  while (comp->isActive()) {
    llm::Chunk chunk = comp->nextChunk();
    std::string nextToken = chunk.getText();

    if (onTokenCallback) onTokenCallback(nextToken);
    answer += nextToken;
    ++numToken;

    if ((!stopSeq.empty()) && (answer.find(stopSeq) != std::string::npos)) break;
  }
  output.numAnswerTokens = numToken;
  output.answerDuration = lut::now() - t0;

  answer = lut::trim(answer);
  output.answer = answer;

  QA qa;
  qa.question = question;
  qa.answer = answer;
  _history.emplace_back(qa);

  return output;
}

}  // namespace libllm
