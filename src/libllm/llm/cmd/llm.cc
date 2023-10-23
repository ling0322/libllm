#include <stdio.h>
#include <string>
#include "lyutil/time.h"
#include "llm/api/llm.h"
#include "llm/cmd/llm_chatglm2.h"

int findOption(int argc, char **argv, const std::string &option) {
  for (int i = 0; i < argc; ++i) {
    if (option == argv[i])
      return i;
  }

  return -1;
}

int main(int argc, char **argv) {
  int iniIdx = findOption(argc, argv, "--ini");
  if (iniIdx == -1 || iniIdx == argc - 1) {
    puts("Usage: llm --ini <model-ini>");
    return 1;
  }

  llm::init();

  llm::Model model = llm::Model::create(argv[iniIdx + 1]);
  llm_chatglm2_main(model);

  llm::destroy();
}
