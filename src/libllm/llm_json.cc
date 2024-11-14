// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <string>

#include "../../third_party/nlohmann/json.hpp"
#include "libllm/llm.h"
#include "lutil/log.h"

using json = nlohmann::json;

thread_local std::string gJsonString;
thread_local std::string gJsonErrorMessage;

struct llm_json_t {
  json jsonObject;
};

llm_json_t *llm_json_parse(const char *json_string) {
  json jsonObject = json::parse(json_string);

  llm_json_t *llmJson = new llm_json_t();
  llmJson->jsonObject = std::move(jsonObject);

  return llmJson;
}

const char *llm_json_dump(llm_json_t *json) {
  CHECK(json) << "json is nullptr";

  gJsonString = json->jsonObject.dump();
  return gJsonString.c_str();
}

void llm_json_delete(llm_json_t *json) {
  delete json;
}

const char *llm_json_get_last_error_message() {
  return gJsonErrorMessage.c_str();
}
