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

#include <stdio.h>
#include <iostream>
#include <string>
#include "lyutil/error.h"
#include "lyutil/flags.h"
#include "lyutil/time.h"
#include "lyutil/strings.h"
#include "llm/api/llm.h"
#include "llm/cli/dialog_manager.h"

using libllm::cli::ChatOutput;
using libllm::cli::DialogManager;
using libllm::cli::PromptBulder;

/// @brief Print the chat statistical data.
/// @param chatOutput Output from chat.
void printChatStat(const ChatOutput &chatOutput) {
  double msPerToken = 1000 * chatOutput.answerDuration / chatOutput.numAnswerTokens;
  std::cout << std::endl << lut::sprintf(
      "(%d token, time=%.2fs, %.2fms per token)",
      chatOutput.numAnswerTokens,
      chatOutput.answerDuration,
      msPerToken) << std::endl;
}

int main(int argc, char **argv) {
  std::string configPath;
  std::string deviceType = "auto";

  const char *usage = 
      "Command line interface for libllm.\n"
      "Usage: llm -config <libllm-config-file> [-device (cpu|gpu|cuda)]";

  lut::Flags flags(usage);
  flags.define("-config", &configPath, "filename of libllm config file.");
  flags.define("-device", &deviceType, "device of the model. (cpu|cuda|auto)");
  flags.parse(argc, argv);

  if (configPath.empty()) {
    flags.printUsage();
    return 1;
  }

  llm::DeviceType device = llm::DeviceType::AUTO;
  if (deviceType == "auto")
    device = llm::DeviceType::AUTO;
  else if (deviceType == "cuda")
    device = llm::DeviceType::CUDA;
  else if (deviceType == "cpu")
    device = llm::DeviceType::CPU;
  else {
    printf("invalid device");
    return 1;
  }
  
  llm::init();
  std::shared_ptr<llm::Model> model = llm::Model::create(configPath, device);
  std::shared_ptr<PromptBulder> promptBuilder = PromptBulder::create(model->getName());
  DialogManager dialogManager(model, promptBuilder);
  for (; ; ) {
    std::string query;
  
    std::cout << "> ";
    std::getline(std::cin, query);
    if (lut::trim(query) == "")
      continue;
    if (lut::trim(query) == "q")
      break;

    ChatOutput chatOutput = dialogManager.chat(query, [](const std::string &token) {
      std::cout << token;
      std::cout.flush();
    });

    printChatStat(chatOutput);
  }
}
