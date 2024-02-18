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

#include "libllm/lut/zip_file.h"

#include <stdlib.h>
#include <memory>
#include "libllm/lut/error.h"
#include "libllm/lut/reader.h"
#include "miniz/miniz.h"

namespace lut {

class ZipFile::Impl {
 public:
  Impl();
  ~Impl();

  void init(const std::string &zip_file);
  bool hasFile(const std::string &filename);
  std::shared_ptr<Reader> open(const std::string &filename);

 private:
  mz_zip_archive _zipArchive;
  std::string _filename;
};

// reader for a file inside the zip file.
class ZipFile::FileReader : public Reader {
 public:
  FileReader();
  ~FileReader();

  void init(std::shared_ptr<ZipFile::Impl> zipFile, const std::string &filename);

 protected:
  // implements interface Reader
  int64_t readData(Span<int8_t> buffer) override;

 private:
  // We need to hold a pointer to the zip file to prevent it from being destructed before the file
  // reader.
  std::shared_ptr<ZipFile::Impl> _zipFile;
  mz_zip_reader_extract_iter_state *_iterState;
};

// -----------------------------------------------------------------------------------------------+
// class ZipFile::Impl                                                                            |
// -----------------------------------------------------------------------------------------------+

ZipFile::Impl::Impl() {
  mz_zip_zero_struct(&_zipArchive);
}

ZipFile::Impl::~Impl() {
  if (_zipArchive.m_pState) {
    mz_zip_reader_end(&_zipArchive);
  }
}

void ZipFile::Impl::init(const std::string &zip_file) {
  _filename = zip_file;

  mz_bool success = mz_zip_reader_init_file(&_zipArchive, zip_file.c_str(), 0);
  if (success = MZ_FALSE) {
    THROW(Aborted, "unable to open zip file: " + zip_file);
  }
}

bool ZipFile::Impl::hasFile(const std::string &filename) {
  int fileIndex = mz_zip_reader_locate_file(&_zipArchive, filename.c_str(), nullptr, 0);
  return fileIndex >= 0;
}

// -----------------------------------------------------------------------------------------------+
// class ZipFile::FileReader                                                                      |
// -----------------------------------------------------------------------------------------------+

ZipFile::FileReader::FileReader() : _iterState(nullptr) {}

ZipFile::FileReader::~FileReader() {
  if (_iterState) {
    mz_zip_reader_extract_iter_free(_iterState);
  }
}

void ZipFile::FileReader::init(std::shared_ptr<ZipFile::Impl> zipFile,
                               const std::string &filename) {
  
}

} // namespace lut
