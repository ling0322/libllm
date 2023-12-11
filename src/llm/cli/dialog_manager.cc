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

llm::Prompt ChatGLM2PromptBuilder::buildPrompt(lut::Span<const QA> history,
                                               const std::string &question) {
  std::string prompt;

  int round = 1;
  for (const QA &qa : history) {
    prompt += lut::sprintf("[Round %d]\n\n问：%s\n\n答：%s\n\n", round, qa.question, qa.answer);
    ++round;
  }
  prompt += lut::sprintf("[Round %d]\n\n问：%s\n\n答：", round, question);

  llm::Prompt hPrompt;
  hPrompt.appendText(prompt);
  return hPrompt;
}

std::string ChatGLM2PromptBuilder::getStopSeq() {
  return "";
}

llm::Prompt ChatGLM3PromptBuilder::buildPrompt(lut::Span<const QA> history,
                                               const std::string &question) {
  llm::Prompt prompt;
  prompt.appendSpecialToken("<|system|>");
  prompt.appendText("\nYou are ChatGLM3, a large language model trained by Zhipu.AI. "
                    "Follow the user's instructions carefully. Respond using markdown.\n");

  for (const QA &qa : history) {
    prompt.appendSpecialToken("<|user|>");
    prompt.appendText(lut::sprintf("\n%s\n", qa.question));
    prompt.appendSpecialToken("<|assistant|>");
    prompt.appendText(lut::sprintf("\n%s\n", qa.answer));
  }

  prompt.appendSpecialToken("<|user|>");
  prompt.appendText(lut::sprintf("\n%s\n", question));
  prompt.appendSpecialToken("<|assistant|>");

  return prompt;
}

std::string ChatGLM3PromptBuilder::getStopSeq() {
  return "";
}

const char *llama2Prompt = 
    "Answer following questions:\n"
    "\n"
    "Q: What's the Capital city of Washinton state?\n"
    "A: Olympia.\n"
    "\n"
    "Q: Who is the 31th president in US?\n"
    "A: Herbert Clark Hoover.\n"
    "\n"
    "Q: %s\n"
    "A: ";

llm::Prompt LlamaPromptBuilder::buildPrompt(lut::Span<const QA> history,
                                            const std::string &question) {
  llm::Prompt prompt;
  prompt.appendText(lut::sprintf(llama2Prompt, question));

  return prompt;
}

std::string LlamaPromptBuilder::getStopSeq() {
  return "\n";
}

std::shared_ptr<PromptBulder> PromptBulder::create(const std::string &modelName) {
  if (modelName == "llama") return std::make_shared<LlamaPromptBuilder>();
  if (modelName == "chatglm2") return std::make_shared<ChatGLM2PromptBuilder>();
  if (modelName == "chatglm3") return std::make_shared<ChatGLM3PromptBuilder>();

  throw lut::AbortedError("unexpected model name: " + modelName);
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

  llm::Prompt prompt = _promptBuilder->buildPrompt(_history, question);
  llm::CompletionConfig config;
  config.setTopK(1);

  double t0 = lut::now();
  llm::Completion comp = _model->complete(prompt);
  output.promptDuration = lut::now() - t0;

  std::string answer;
  std::string stopSeq = _promptBuilder->getStopSeq();
  t0 = lut::now();
  int numToken = 0;
  while (comp.isActive()) {
    llm::Chunk chunk = comp.nextChunk();
    std::string nextToken = chunk.getText();

    if (onTokenCallback) onTokenCallback(nextToken);
    answer += nextToken;
    ++numToken;

    if ((!stopSeq.empty()) && (answer.find(stopSeq) != std::string::npos))
      break;
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


}  // namespace cli
}  // namespace libllm
