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

#pragma once

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define LUT_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define LUT_HAS_CPP_ATTRIBUTE(x) 0
#endif

#if LUT_HAS_CPP_ATTRIBUTE(clang::lifetimebound)
#define LUT_LIFETIME_BOUND [[clang::lifetimebound]]
#else
#define LUT_LIFETIME_BOUND
#endif

#ifdef __APPLE__
#define LUT_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define LUT_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || \
      defined(_WIN32) || defined(__MINGW32__)
#define LUT_PLATFORM_WINDOWS
#else
#error unknown platform
#endif

#if defined(__clang__)
#define LUT_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
#define LUT_COMPILER_GCC
#elif defined(_MSC_VER)
#define LUT_COMPILER_MSVC
#else
#error unknown compiler
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define LUT_CHECK_RETURN [[nodiscard]]
#elif defined(LUT_COMPILER_MSVC)
#define LUT_CHECK_RETURN _Check_return_
#elif defined(LUT_COMPILER_GCC)
#define LUT_CHECK_RETURN __attribute__((warn_unused_result))
#else
#define LUT_CHECK_RETURN
#endif

#if (defined(_M_AMD64) || defined(__amd64__) || defined(__x86_64__))
#define LUT_ARCH_AMD64
#elif defined(_M_ARM64) || defined(__aarch64__)
#define LUT_ARCH_AARCH64
#else
#error unknown CPU architecture
#endif
