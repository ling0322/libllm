#include "libllm/llm.h"

#include <string.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include "../../third_party/nlohmann/json.hpp"
#include "libllm/context.h"
#include "libllm/dtype.h"
#include "libllm/functional.h"
#include "libllm/generator.h"
#include "libllm/model_for_generation.h"
#include "libllm/operators.h"
#include "libllm/prompt.h"
#include "libllm/tokenizer.h"
#include "libllm/whisper.h"
#include "libllm/whisper_decoder.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "lutil/zip_file.h"

using libllm::Context;
using libllm::GenerationConfig;
using libllm::Generator;
using libllm::LongType;
using libllm::ModelForGeneration;
using libllm::Prompt;
using libllm::Tokenizer;
using libllm::whisper::RecognitionResult;
using libllm::whisper::WhisperDecoder;
using libllm::whisper::WhisperModel;
using lut::IniConfig;
using json = nlohmann::json;

std::once_flag gLlmInitOnce;

thread_local std::string gJsonString;
thread_local std::string gJsonErrorMessage;

constexpr char LlmConfigKey_GeneratorType[] = "generator.type";
constexpr char LlmConfigKey_WhisperLang[] = "whisper.language";
constexpr char LlmConfigValue_Sampler[] = "sampler";
constexpr char LlmConfigValue_Whisper[] = "whisper";

struct llm_model_impl_t {
  std::shared_ptr<ModelForGeneration> model;
  std::shared_ptr<Tokenizer> tokenizer;
};

struct llm_completion_impl_t {
  std::weak_ptr<ModelForGeneration> model_for_generation;
  std::shared_ptr<Generator> generator;
};

struct llm_json_impl_t {
  json jsonObject;
};

struct llm_asr_model_impl_t {
  std::shared_ptr<WhisperModel> model;
};

struct llm_asr_recognition_impl_t {
  std::shared_ptr<WhisperDecoder> decoder;
};

namespace libllm {
namespace api {

thread_local int gErrorCode = static_cast<int>(lut::ErrorCode::OK);
thread_local char gErrorMessage[512] = "";
static std::atomic<bool> gInitialized{false};

void llmSetErrorMessage(const std::string &message) {
  std::string what = message;
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

void checkJsonKeys(
    const json &json,
    std::initializer_list<std::pair<std::string_view, bool>> schema) {
  std::set<std::string_view> keys;
  for (auto &[key, value] : json.items()) {
    keys.emplace(key);
  }

  for (const auto &entry : schema) {
    std::string_view key = entry.first;
    bool required = entry.second;

    auto it = keys.find(key);
    if (required && it == keys.end()) {
      throw lut::AbortedError(lut::sprintf("json: required key \"%s\" not found", key));
    }

    if (it != keys.end()) keys.erase(it);
  }

  if (!keys.empty()) {
    throw lut::AbortedError(lut::sprintf("json: unexpected key \"%s\"", *keys.begin()));
  }
}

Prompt buildPromptFromJson(const ModelForGeneration *model, const json &kwargsJson) {
  const json &messageJsons = kwargsJson["messages"];
  std::vector<Message> messages;
  for (const json &messageJson : messageJsons) {
    Message message;
    message.role = messageJson["role"];
    message.content = messageJson["content"];

    messages.emplace_back(message);
    if (messageJson.size() != 2) {
      throw lut::AbortedError("invalid json for message");
    }
  }

  return model->buildPrompt(messages);
}

template<typename T>
T getValueFromJson(const json &j, std::string_view key, T defaultVal) {
  T val = defaultVal;
  if (kwargsJson.contains(key)) {
    val = kwargsJson[key];
  }

  return val;
}

GenerationConfig parseGenerationConfig(const json &kwargsJson) {
  GenerationConfig config;
  config.temperature = getValueFromJson<float>(kwargsJson, "temperature", 1.0);
  config.topK = getValueFromJson<int>(kwargsJson, "top_k", 50);
  config.topP = getValueFromJson<float>(kwargsJson, "top_p", 0.8);

  return config;
}

int32_t llmErrorSetInvalidArg(const std::string &argName) {
  llmSetErrorMessage("invalid argument: " + argName);
  return LLM_ERROR_INVALID_ARG;
}

int32_t llmErrorSetAborted(const std::string &what) {
  llmSetErrorMessage(what);
  return LLM_ERROR_ABORTED;
}

int32_t llmErrorSetInsufficientBuffer() {
  llmSetErrorMessage("Insufficient buffer size.");
  return LLM_ERROR_INSUFFICIENT_BUFFER;
}

int32_t llmErrorSetEOF() {
  llmSetErrorMessage("End of file.");
  return LLM_ERROR_EOF;
}

libllm::Device parseDevice(const std::string &device) {
  if (device == "cpu") {
    return libllm::Device::getCpu();
  } else if (device == "cuda") {
    return libllm::Device::getCuda();
  } else if (device == "auto") {
    if (Device::isCudaAvailable()) {
      return Device::getCuda();
    } else {
      return Device::getCpu();
    }
  } else {
    throw lut::AbortedError("invalid device: " + device);
  }
}

}  // namespace api
}  // namespace libllm

// -- api implementation ----------

using namespace libllm;
using namespace libllm::api;

void llm_init() {
  std::call_once(gLlmInitOnce, []() {
    try {
      lut::setLogLevel(lut::LogSeverity::kINFO);
      libllm::initOperators();
    } catch (const lut::Error &e) {
      LOG(ERROR) << "initialize libllm failed: " << e.what();
    }
  });
}

const char *llm_get_last_error_message() {
  return gErrorMessage;
}

int32_t llm_model_init(llm_model_t *m) {
  *m = new llm_model_impl_t();
  return 0;
}

int32_t llm_model_destroy(llm_model_t *m) {
  if (!m) return llmErrorSetInvalidArg("m");

  delete *m;
  *m = nullptr;
  return 0;
}

int32_t llm_model_load(llm_model_t *m, llm_json_t *kwargs) {
  try {
    libllm::Device device;
    std::shared_ptr<lut::ZipFile> package;
    json object = (*kwargs)->jsonObject;
    for (auto &[key, value] : object.items()) {
      if (key == "filename") {
        package = lut::ZipFile::fromFile(value);
      } else if (key == "device") {
        device = parseDevice(value);
      } else {
        return llmErrorSetAborted("invalid key in options: " + key);
      }
    }

    if (!package) return llmErrorSetAborted("options.filename undefined");
    if (device.getType() == libllm::Device::kUnknown) {
      return llmErrorSetAborted("options.device undefined");
    }

    Context ctx;
    ctx.setDevice(device);
    ctx.setFloatDType(F::getDefaultFloatType(device));
    std::shared_ptr<ModelForGeneration> model = ModelForGeneration::fromPackage(ctx, package.get());

    (*m)->model = model;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_model_get_info(llm_model_t *m, llm_json_t *info) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!info) return llmErrorSetInvalidArg("info");

  try {
    json infoJson;
    infoJson["name"] = (*m)->model->getName();
    (*info)->jsonObject = infoJson;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_completion_init(llm_completion_t *c) {
  *c = new llm_completion_impl_t();
  return 0;
}

int32_t llm_completion_destroy(llm_completion_t *c) {
  if (!c) return llmErrorSetInvalidArg("c");
  delete *c;
  *c = nullptr;

  return 0;
}

int32_t llm_model_complete(llm_model_t *m, llm_json_t *kwargs, llm_completion_t *comp) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!kwargs) return llmErrorSetInvalidArg("kwargs");
  if (!comp) return llmErrorSetInvalidArg("comp");

  try {
    const json &kwargsJson = (*kwargs)->jsonObject;
    checkJsonKeys(
        kwargsJson,
        {{"temperature", false}, {"top_k", false}, {"top_p", false}, {"messages", true}});

    GenerationConfig config = parseGenerationConfig(kwargsJson);
    (*comp)->generator = SamplingGenerator::newGenerator(config, (*m)->model);
    (*comp)->model_for_generation = (*m)->model;

    Prompt prompt = buildPromptFromJson((*m)->model.get(), kwargsJson);
    (*comp)->generator->setPrompt(prompt);
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_completion_get_next_chunk(llm_completion_t *c, llm_json_t *chunk) {
  if (!c) return llmErrorSetInvalidArg("c");
  if (!chunk) return llmErrorSetInvalidArg("chunk");

  try {
    bool ok = (*c)->generator->generate();
    if (!ok) {
      return llmErrorSetEOF();
    }

    json chunkJson;
    chunkJson["text"] = (*c)->generator->getToken();
    (*chunk)->jsonObject = chunkJson;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_json_init(llm_json_t *j) {
  llm_json_impl_t *llmJson = new llm_json_impl_t();
  *j = llmJson;

  return 0;
}

int32_t llm_json_destroy(llm_json_t *j) {
  delete *j;
  *j = nullptr;

  return 0;
}

int32_t llm_json_parse(llm_json_t *j, const char *json_str) {
  if (!j) return llmErrorSetInvalidArg("j");
  if (!json_str) return llmErrorSetInvalidArg("json_str");

  try {
    (*j)->jsonObject = json::parse(json_str);
  } catch (lut::Error &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size) {
  if (!j) return llmErrorSetInvalidArg("j");
  if (!buf) return llmErrorSetInvalidArg("buf");
  if (buf_size <= 0) return llmErrorSetInvalidArg("buf_size");

  std::string jsonStr = (*j)->jsonObject.dump();
  if (jsonStr.size() >= buf_size) {
    return llmErrorSetInsufficientBuffer();
  }

  snprintf(buf, buf_size, "%s", jsonStr.c_str());
  return 0;
}

int32_t llm_asr_model_init(llm_asr_model_t *m) {
  *m = new llm_asr_model_impl_t();
  return 0;
}

int32_t llm_asr_model_load(llm_asr_model_t *m, llm_json_t *options) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!options) return llmErrorSetInvalidArg("options");

  try {
    json object = (*options)->jsonObject;
    checkJsonKeys(object, {{"filename", true}, {"device", true}});

    std::shared_ptr<lut::ZipFile> package = lut::ZipFile::fromFile(object["filename"]);
    libllm::Device device = parseDevice(object["device"]);

    Context ctx = Context().withName("whisper");
    ctx.setDevice(device);
    ctx.setFloatDType(F::getDefaultFloatType(device));
    std::shared_ptr<WhisperModel> whisperModel = WhisperModel::fromPackage(ctx, package.get());

    (*m)->model = whisperModel;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_asr_model_destroy(llm_asr_model_t *m) {
  delete *m;
  *m = nullptr;

  return 0;
}

int32_t llm_asr_recognition_init(llm_asr_recognition_t *r) {
  *r = new llm_asr_recognition_impl_t();
  return 0;
}

int32_t llm_asr_recognition_destroy(llm_asr_recognition_t *r) {
  delete *r;
  *r = nullptr;

  return 0;
}

int32_t llm_asr_recognize_media_file(
    llm_asr_model_t *model,
    llm_json_t *options,
    llm_asr_recognition_t *recognition) {
  if (!recognition) return llmErrorSetInvalidArg("r");
  if ((!model) || !(*model)->model) return llmErrorSetInvalidArg("model");
  if (!options) return llmErrorSetInvalidArg("options");

  try {
    std::string mediaFile;
    json object = (*options)->jsonObject;
    for (auto &[key, value] : object.items()) {
      if (key == "media_file") {
        mediaFile = value;
      } else {
        throw lut::AbortedError("invalid key in options: " + key);
      }
    }

    std::shared_ptr<WaveStream> stream = FFmpegWaveStream::open(mediaFile);
    std::shared_ptr<WhisperModel> whisperModel = (*model)->model;
    std::shared_ptr<Wave> wave = std::make_shared<Wave>(stream);
    std::shared_ptr<WhisperDecoder> decoder = WhisperDecoder::create(whisperModel, wave);

    (*recognition)->decoder = decoder;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_asr_recognition_get_next_result(llm_asr_recognition_t *r, llm_json_t *result) {
  if ((!r) || !(*r)->decoder) return llmErrorSetInvalidArg("r");
  if (!result) return llmErrorSetInvalidArg("result");

  try {
    std::optional<RecognitionResult> recoResult = (*r)->decoder->nextResult();
    if (recoResult) {
      json resultJson;
      resultJson["text"] = recoResult->text;
      resultJson["language"] = recoResult->language;
      resultJson["begin"] = recoResult->begin.totalNanoseconds() / 1000000;
      resultJson["end"] = recoResult->end.totalNanoseconds() / 1000000;

      (*result)->jsonObject = resultJson;
    } else {
      return llmErrorSetEOF();
    }
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}
