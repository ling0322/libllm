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

#include "llm/cli/dialog_manager.h"

#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "lyutil/time.h"

namespace libllm {
namespace cli {

ChatOutput::ChatOutput() : 
    numAnswerTokens(0),
    promptDuration(0.0),
    answerDuration(0.0) {}

std::string ChatGLM2PromptBuilder::buildPrompt(
    ly::Span<const QA> history,
    const std::string &question) {
  std::string prompt;

  int round = 1;
  for (const QA &qa : history) {
    prompt += ly::sprintf("[Round %d]\n\n问：%s\n\n答：%s\n\n", round, qa.question, qa.answer);
    ++round;
  }
  prompt += ly::sprintf("[Round %d]\n\n问：%s\n\n答：", round, question);

  return prompt;
}

std::string LlamaPromptBuilder::buildPrompt(
    ly::Span<const QA> history,
    const std::string &question) {
  std::string prompt;
  for (const QA &qa : history) {
    prompt += ly::sprintf("QUESTION:\n%s\n\nANSWER:\n%s\n\n", qa.question, qa.answer);
  }
  prompt += ly::sprintf("QUESTION:\n%s\n\nANSWER:\n", question);

  return prompt;
}

std::shared_ptr<PromptBulder> PromptBulder::create(const std::string &modelName) {
  if (modelName == "llama") return std::make_shared<LlamaPromptBuilder>();
  if (modelName == "chatglm2") return std::make_shared<ChatGLM2PromptBuilder>();

  throw ly::AbortedError("unexpected model name: " + modelName);
  return nullptr;
}

DialogManager::DialogManager(std::shared_ptr<llm::Model> model,
                             std::shared_ptr<PromptBulder> promptBuilder) :
    _model(model),
    _promptBuilder(promptBuilder) {}

ChatOutput DialogManager::chat(
    const std::string &question,
    std::function<void(const std::string &)> onTokenCallback) {
  ChatOutput output;

  std::string prompt = _promptBuilder->buildPrompt(_history, question);
  llm::CompletionConfig config;
  config.setTopK(1);

  double t0 = ly::now();
  llm::Completion comp = _model->complete(prompt);
  output.promptDuration = ly::now() - t0;

  std::string answer;
  t0 = ly::now();
  int numToken = 0;
  while (!comp.stopped()) {
    llm::Chunk chunk = comp.nextChunk();
    std::string nextToken = chunk.getText();

    if (onTokenCallback) onTokenCallback(nextToken);
    answer += nextToken;
    ++numToken;
  }
  output.numAnswerTokens = numToken;
  output.answerDuration = ly::now() - t0;

  answer = ly::trim(answer);
  output.answer = answer;

  QA qa;
  qa.question = question;
  qa.answer = answer;
  _history.emplace_back(qa);

  return output;
}


}  // namespace cli
}  // namespace libllm
