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

#include "lynn/cpu/log_mel_spectrogram.h"

#include "catch2/catch_amalgamated.hpp"
#include "lutil/base64.h"
#include "lynn/cpu/fingerprint.h"
#include "lynn/functional.h"
#include "lynn/wave.h"

namespace libllm {
namespace op {
namespace cpu {

const char *gDataEnUsHelloBase64 =
    "UklGRnwWAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YVgWAACH/0L/gv+W/5D/dP+K/zX/MP5m/c/9DP+a/"
    "1X/TP8dAIwACwA7/6f+fP63/nj/y/8K/1z+6f4v/wD+p/wD/Wr+Zv+F/7z+Av6o/T/+7P7i/cf8D/7R/38AGwADAGH/"
    "Av7V/aL+Gv60/J78zv2V/j7+3/3i/Tj+p/49/lD98f0J/4n/SADmAAQAj/4p/f/8wP0M/s791/0Q/nr9rPwT/f39S/"
    "23+3T83P/JAG3/UP/L/2D/2v4M/8b/3P7t/VD+ev29+9j5KPiB93X5Dv1l/8L/FgGtAtYCLQFS/279v/u2+3T8e/"
    "zp+6z87f36/f79GP/b/ygBrQLZAjIC5QAG/6T92vtJ+5v8Vf2g/TD+jP/IAaoCoQG4ATYDPgS3AgAA6v5L/o38mvss/"
    "KD9G/9R//L+0P4k/pD9B/6+/vj+5/4R/0P/Vv9r/p3+NgA4AQ4BUwE+AmgCDwEEAKkA5ABY/4b+Rv+E/13/"
    "MgCPAVgBgQDEALICAwPeAREC8wIvA/YB5QBtAX0CmAKbAmMCaQHa/9H/fwCA/9b+dwCIAuABu//K/mL+Mf2a/E/"
    "+AQCoAI8BsgKCAtb/Lv5t/1EAXf/4/s//kwB6AFP/Fv5L/"
    "Sf+ZgCQAfsBewJFA3EEGgTTAfj+Jf6tAIQCxQL8AoUD8wMIBPMDSARgA3EE2QZ6CDsIogVxBKgCNAC7/Y39v/7T/"
    "wkBqgHCAtwB7QCEAfMBRQEQAA8AAQDZ/pr8kfs//KX7zPtu/YT+f/7E/RH9wft4+kf76/ub+6D88/3D/h3+T/1g/"
    "YT9jf5jAHgCUwRVBsEHygh8CbwJUwkZCAUIcQiCCGIJkgvlDAQNhw00DoEMJArxCA4IVwclBt4FOAQYAjYCnAFz/"
    "hj6WPje90r2N/Un9rb3n/ib+dn59PhC+MP4G/p/+tT5Zfkh+f/4dfhX9/P2ZPfn93H4Rvm2+g/"
    "9BgC9A8sGtgmzDjgSihDeCtUGAge4CH0LIQ6SEdwV/xgwGu8X2xRRExoScBDpDgsMWwT++o/"
    "ypuo45cnje+bc7MH0kvxpAEEBHALKATcBUwCJ/x4AxAC5/"
    "7n7ZvVD8ALt7uox6ifr+u6g9Oz63v+QAZ8BjgKxAycF3gbICVUO0BIkFgoV9g+"
    "GC2AK4woDDLwOyBT5Gj8eGx8aG5kVxBDnDUMJ+/8R9jvvZO0V69fobugg7Fv0+Pq7/"
    "iEAtgHTA0sEhwNEAk4C2AMyA5wAfPuT9WTyou9W7WXrWevI7g/"
    "0HPgJ+6H8rP22AIwCXAOABFUG4glEDeoOWhDfEc0RNg87CzgJ6QnsDIUQrRRdGGYaVxtrGR8VUBDxCw0HH/"
    "8o9WTvau6q7UHtg+0c8Av1fPi3+uX7jfwL/8IB0wLqAoEC+QJTAuH+YfrE9VrzNvJc8Abv8u6p8BfzBvUD96D4RPoJ/YH/"
    "9ADOAt0FPwlhCxsMzAxSDroPFBCkDtoMcw0wD0MRPxOQFbwXohgxGCMWWxKXCxADJvkv8HLr4+h36Lrqiu6J8i/"
    "2NPiH+HT4LPkD+xz8pf1vAJsD/wQ6BC0BVvxj+MD0m/Fk7z/unO/v8o31p/"
    "Z29mX2sffq+P75GPyAAB8GpwrEDMQNCw89EM4Paw1uC9ELcQ4IEVMT8BXYF18ZQRkTFogSHA4nB439o/"
    "NY7hvs9+pZ6xrt7u8U82z1uPYE91T3Nfnh+0H+OADeArEF3galBdsCQgCX/m78t/hx9Qf0s/"
    "Mw84XyofGw8aDzCvYv+Ef6m/"
    "0zAgAGkghiCgEMBQ5MELAREBGpD4kPkBAqERkRcBILFdwXphleGcYXFhXpEFMKJQAm9ofwDO5K7ELsP+7W8RD2tPj2+"
    "Xb5O/nM+kb87/0yAG8D4wY+CXEIqQWOAiv/jfun97z0NfMT87LzP/Xy9Sf2t/"
    "do+oP8uv1BAJQDUwelCs4MiQ4UEH8RShKBETEQ/Q+HEOMRPBQFF/"
    "oZqxzcHQQdTRltE9YKfv+29ZHuReqO6KboV+ta773yjPX49uv2Vfdj9034lvqf/f0BxwWbCFwJugdtBaoC+f2o+F/"
    "1yPPt8mPyEvP89OD28/ck+e75xfq0/O/"
    "+GwJ0BdIIaQxPD9IQ+xBhEGYPcA4PDgIPXxHZFP4XYBphGycafReqEVIJbv9M9XXtjOi75mznTOov7h/yYfRA9d70k/"
    "TN9Af1bvYZ+cP9UgJLBYwGTwbVBFgC3v59+oP25fOM8ofxe/AB8VbzP/"
    "Vs9kj3Jfjx+en7+f2vALcDbQfzCl0Nqw59D3MP4g43DrcNwA4iEf8TTBYPGPoYcBj/"
    "FFQOwQQy+qPxlesE53vl+uZg6lHu1vCD8jPz8PLf8sHyJ/OD9V75Ov6YAg0F4QW7BRcECwAx+wz3RvTG8r3xCfK288j15/"
    "fc+Nf47fht+fv6ovzM/lUCkAZqCtcM/g1jDt4NRA3HDPoMEQ5RENMSgBUHGAkZ2BhsFvQPmQZt/"
    "J7zPO1P6JDmueet6nHukfFw88rzlvNN87Dy8/Kr9Ab43vwrAY0EsQZsBwUHpgT9/9v6aven9MjygPIm84f1A/iE+Yr6m/"
    "oU+wH81/xR/vYASwQgCGALKw2ZDUgO/g71Du4O1Q7pD9oR4BN9FsQXXRfDFRAR3QmeADn4PPKH7K7pduku68buR/"
    "HG83H1KvVI9V30iPNe9Dn2uPkE/q8BZAQeBvgGDgb3AtD+QPs6+B/2PfWM9Sv39fjQ+lH8qfwq/ar9HP5L/"
    "8oAiwLqBLAHzQnZCsULSwzfDE8NlA3/Dj8QJBKDFKwVPBbTFdITnA+lCEgBYvsA9svxTO/A7sfv6/B88tfzO/So9L30D/"
    "RP9L/0T/Yx+c37H/6CANsCYwQMBGEB3P6m/Hr5V/cT9i/2ZPiG+gv9sv/EAOgBfQLXAaUBLgJnA/"
    "kFKwgICooMVg6gD6cQkhFWEnoTORVtFqIWTBalE1AObwfR//L4lvKm7b7qFOpI63ftpe8A8XXyV/"
    "Sl9DD1dfak93H67P3SAEgDoAWyBwYIwAT+AAT79/OH8TPvI+7l7xzz1PnY/"
    "gYBhQMlBX4GAgaLBWwHcQmoDIsQ1RLDFbIY5xlZG1kbuhpmG3YaBRkdFOgK1QEg96ftNudg4ETdxt9x5H3p6e1S8ov26Pk"
    "P+5z75/w9/zMDwwbzCIcKPgvVC2kIRwAj+Vfymep45Y7j+eQj6orvsfaZ/"
    "WcBNQUfB8YHjQjhCK0Lmg9VE4QW1Rn9HFsdNh0bHlYdmRuiGiMZ3hbvD68ChPa57DDi7tjK1NjWo9vV4sTrHPRs+oD/"
    "0wRABigG1AbfB2sKgwu9C1ALqAhPBb3+SfVM7iflVtxX3F/e9eKX6uLxa/xDBKAHMwvdC+oLrQ3gD/0SdRaCGbscSB/"
    "hHUgdYBsUGnsa2xifGDcVzA96A+Hwk+XU2w7SE8+b0SzbYuet8tH8KAPtB5sKJAr5B9IHWwljC80NHg3DCQQFX/"
    "8N97Dr9eHb3NLZPtk83yLmWO77+KwB8AcSCgQLnA17DooNBA8GE70W9Bl4G9YaShpAGTwXdxTDEzAVAhbnFU0Skwh6+"
    "MnpseDC1orQGdMO20LnVfNv/uIFrgmvDAsNMwplB68HIAl+CpEKEQjgA23+7/bQ7FHjRtyC1wfW1dmt4r7sM/"
    "cqAnsKzA54EBIR0xCfD38PzRHLFJIX1Rm4GhoaihhCFuITsBJHEvQS5RN+"
    "EE8HqfmN6wPjzdpQ0zvU99x66gD2vP5YB94LDw3eCxYIkQUWBgUICgkvCKMFjgH9+"
    "1H0MOoQ4Y7bq9rL3EzfBOVb71v6AgShCjwNkw+"
    "8EY0S8xGzD7sPHBTaFoIW0RV8FREWLhZDFEwSNxJNFDEWOBNqB2H4jOxj4dbYZtMQ1AHdSukG960BHQfxC84OcAwjCG8Fq"
    "wUQB+YI8AhgBOz+L/"
    "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";

CATCH_TEST_CASE("test logMelSpectrogram", "[op][cpu][logmelspectrogram]") {
  std::vector<Byte> pcmData = lut::decodeBase64(gDataEnUsHelloBase64);
  CATCH_REQUIRE(std::string(reinterpret_cast<const char *>(pcmData.data()) + 36, 4) == "data");

  lut::Span<const Byte> pcmSpan(pcmData);
  pcmSpan = pcmSpan.subspan(44);  // 44: pcm header size.

  Tensor wave = Wave::toTensor(pcmSpan);
  Tensor features = logMelSpectrogram(wave);

  CATCH_REQUIRE(
      F::allClose(
          op::cpu::fingerprint(features),
          Tensor::create<float>(
              {8},
              {0.5365, 0.6787, 0.1886, 0.4008, -0.2633, -0.3035, -0.4268, -0.6635})));
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
