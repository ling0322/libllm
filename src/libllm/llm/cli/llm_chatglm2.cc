#include "llm/cmd/llm_chatglm2.h"
#include "lyutil/time.h"
#include "lyutil/strings.h"

#include <iostream>
#include <string>
#include <vector>

struct QA {
  std::string question;
  std::string answer;
};

class ChatGLM2Dialog {
 public:
  ChatGLM2Dialog(const llm::Model &model);

  void chat(const std::string &query);

 private:
  llm::Model _model;
  std::vector<QA> _history;

  std::string buildPrompt(const std::string &query);
};

ChatGLM2Dialog::ChatGLM2Dialog(const llm::Model &model) : _model(model) {}

void ChatGLM2Dialog::chat(const std::string &query) {
  std::string prompt = buildPrompt(query);
  llm::CompletionConfig config;
  config.setTopK(1);

  llm::Completion comp = _model.complete(prompt);

  std::string answer;
  double t0 = ly::now();
  int numToken = 0;
  while (!comp.stopped()) {
    llm::Chunk chunk = comp.nextChunk();
    answer += chunk.getText();

    printf("%s", chunk.getText().c_str());
    fflush(stdout);
    ++numToken;
  }
  double t1 = ly::now();
  printf("\n(%d token, time=%.2fs, %.2fms per token)\n",
         numToken,
         t1 - t0,
         (t1 - t0) / numToken * 1000 );

  answer = ly::trim(answer);
  QA qa;
  qa.question = query;
  qa.answer = answer;
  _history.emplace_back(qa);
}

std::string ChatGLM2Dialog::buildPrompt(const std::string &query) {
  std::string prompt;

  int round = 1;
  for (const QA &qa : _history) {
    prompt += ly::sprintf("[Round %d]\n\n问：%s\n\n答：%s\n\n", round, qa.question, qa.answer);
    ++round;
  }
  prompt += ly::sprintf("[Round %d]\n\n问：%s\n\n答：", round, query);

  return prompt;
}

void llm_chatglm2_main(const llm::Model &model) {
  ChatGLM2Dialog dialog(model);
  for (; ; ) {
    std::string query;
    if (query == "q")
      break;
  
    std::cout << "> ";
    std::getline(std::cin, query);
    if (ly::trim(query) == "")
      continue;

    dialog.chat(query);
  }
}
