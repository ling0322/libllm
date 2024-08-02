// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// This file is modified from StoreZipReader in Tencent/ncnn project.
// Original file: https://github.com/Tencent/ncnn/blob/master/tools/pnnx/src/storezip.h

#include "lut/zip_file.h"

#include <stdlib.h>

#include <algorithm>
#include <map>
#include <memory>

#include "lut/error.h"
#include "lut/reader.h"
#include "lut/strings.h"

#ifdef _MSC_VER
#define FSEEK64 _fseeki64
#elif _FILE_OFFSET_BITS == 64
#define FSEEK64 fseeko
#else
#define FSEEK64 fseeko64
#endif  // _MSC_VER

namespace lut {

class ZipFile::Impl {
 public:
  struct StoreZipMeta {
    uint64_t offset;
    uint64_t size;
  };

  Impl();
  ~Impl();

  void open(const std::string &path);
  std::vector<std::string> getList() const;
  StoreZipMeta getFileMeta(const std::string &name) const;
  FILE *getFp();

 private:
  FILE *_fp;
  std::map<std::string, StoreZipMeta> _filemetas;

  int close();
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
  FILE *_fp;
  Impl::StoreZipMeta _fileMeta;
  int64_t _pos;
  std::string _filename;
};

// -----------------------------------------------------------------------------------------------+
// class ZipFile::Impl                                                                            |
// -----------------------------------------------------------------------------------------------+

// https://stackoverflow.com/questions/1537964/visual-c-equivalent-of-gccs-attribute-packed
#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

PACK(struct local_file_header {
  uint16_t version;
  uint16_t flag;
  uint16_t compression;
  uint16_t last_modify_time;
  uint16_t last_modify_date;
  uint32_t crc32;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t file_name_length;
  uint16_t extra_field_length;
});

PACK(struct zip64_extended_extra_field {
  uint64_t uncompressed_size;
  uint64_t compressed_size;
  uint64_t lfh_offset;
  uint32_t disk_number;
});

PACK(struct central_directory_file_header {
  uint16_t version_made;
  uint16_t version;
  uint16_t flag;
  uint16_t compression;
  uint16_t last_modify_time;
  uint16_t last_modify_date;
  uint32_t crc32;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t file_name_length;
  uint16_t extra_field_length;
  uint16_t file_comment_length;
  uint16_t start_disk;
  uint16_t internal_file_attrs;
  uint32_t external_file_attrs;
  uint32_t lfh_offset;
});

PACK(struct zip64_end_of_central_directory_record {
  uint64_t size_of_eocd64_m12;
  uint16_t version_made_by;
  uint16_t version_min_required;
  uint32_t disk_number;
  uint32_t start_disk;
  uint64_t cd_records;
  uint64_t total_cd_records;
  uint64_t cd_size;
  uint64_t cd_offset;
});

PACK(struct zip64_end_of_central_directory_locator {
  uint32_t eocdr64_disk_number;
  uint64_t eocdr64_offset;
  uint32_t disk_count;
});

PACK(struct end_of_central_directory_record {
  uint16_t disk_number;
  uint16_t start_disk;
  uint16_t cd_records;
  uint16_t total_cd_records;
  uint32_t cd_size;
  uint32_t cd_offset;
  uint16_t comment_length;
});

ZipFile::Impl::Impl() {
  _fp = 0;
}

ZipFile::Impl::~Impl() {
  close();
}

void ZipFile::Impl::open(const std::string &path) {
  close();

  _fp = fopen(path.c_str(), "rb");
  if (!_fp) {
    THROW(Aborted, lut::sprintf("unable to open file %s", path));
  }

  while (!feof(_fp)) {
    // peek signature
    uint32_t signature;
    int nread = fread((char *)&signature, sizeof(signature), 1, _fp);
    if (nread != 1) break;

    // fprintf(stderr, "signature = %x\n", signature);

    if (signature == 0x04034b50) {
      local_file_header lfh;
      size_t n = fread((char *)&lfh, sizeof(lfh), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }

      if (lfh.flag & 0x08) {
        THROW(Aborted, "zip file contains data descriptor, this is not supported yet");
      }

      if (lfh.compression != 0 || lfh.compressed_size != lfh.uncompressed_size) {
        THROW(Aborted, lut::sprintf("not stored zip file"));
      }

      // file name
      std::string name;
      name.resize(lfh.file_name_length);
      n = fread((char *)name.data(), name.size(), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }

      uint64_t compressed_size = lfh.compressed_size;
      uint64_t uncompressed_size = lfh.uncompressed_size;
      if (compressed_size == 0xffffffff && uncompressed_size == 0xffffffff) {
        uint16_t extra_offset = 0;
        while (extra_offset < lfh.extra_field_length) {
          uint16_t extra_id;
          uint16_t extra_size;
          size_t n = fread((char *)&extra_id, sizeof(extra_id), 1, _fp);
          if (!n) {
            THROW(Aborted, lut::sprintf("unable to read file %s", path));
          }

          n = fread((char *)&extra_size, sizeof(extra_size), 1, _fp);
          if (!n) {
            THROW(Aborted, lut::sprintf("unable to read file %s", path));
          }

          if (extra_id != 0x0001) {
            // skip this extra field block
            fseek(_fp, extra_size - 4, SEEK_CUR);
            extra_offset += extra_size;
            continue;
          }

          // zip64 extra field
          zip64_extended_extra_field zip64_eef;
          n = fread((char *)&zip64_eef, sizeof(zip64_eef), 1, _fp);
          if (!n) {
            THROW(Aborted, lut::sprintf("unable to read file %s", path));
          }

          compressed_size = zip64_eef.compressed_size;
          uncompressed_size = zip64_eef.uncompressed_size;

          // skip remaining extra field blocks
          fseek(_fp, lfh.extra_field_length - extra_offset - 4 - sizeof(zip64_eef), SEEK_CUR);
          break;
        }
      } else {
        // skip extra field
        fseek(_fp, lfh.extra_field_length, SEEK_CUR);
      }

      StoreZipMeta fm;
      fm.offset = ftell(_fp);
      fm.size = compressed_size;

      _filemetas[name] = fm;

      // fprintf(stderr, "%s = %d  %d\n", name.c_str(), fm.offset, fm.size);

      fseek(_fp, compressed_size, SEEK_CUR);
    } else if (signature == 0x02014b50) {
      central_directory_file_header cdfh;
      size_t n = fread((char *)&cdfh, sizeof(cdfh), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }

      // skip file name
      fseek(_fp, cdfh.file_name_length, SEEK_CUR);

      // skip extra field
      fseek(_fp, cdfh.extra_field_length, SEEK_CUR);

      // skip file comment
      fseek(_fp, cdfh.file_comment_length, SEEK_CUR);
    } else if (signature == 0x06054b50) {
      end_of_central_directory_record eocdr;
      size_t n = fread((char *)&eocdr, sizeof(eocdr), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }

      // skip comment
      fseek(_fp, eocdr.comment_length, SEEK_CUR);
    } else if (signature == 0x06064b50) {
      zip64_end_of_central_directory_record eocdr64;
      size_t n = fread((char *)&eocdr64, sizeof(eocdr64), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }

      // skip comment
      fseek(_fp, eocdr64.size_of_eocd64_m12 - 44, SEEK_CUR);
    } else if (signature == 0x07064b50) {
      zip64_end_of_central_directory_locator eocdl64;
      size_t n = fread((char *)&eocdl64, sizeof(eocdl64), 1, _fp);
      if (!n) {
        THROW(Aborted, lut::sprintf("unable to read file %s", path));
      }
    } else {
      THROW(Aborted, lut::sprintf("unsupported signature %x\n", signature));
    }
  }
}

std::vector<std::string> ZipFile::Impl::getList() const {
  std::vector<std::string> names;
  for (std::map<std::string, StoreZipMeta>::const_iterator it = _filemetas.begin();
       it != _filemetas.end();
       ++it) {
    names.push_back(it->first);
  }

  return names;
}

ZipFile::Impl::StoreZipMeta ZipFile::Impl::getFileMeta(const std::string &name) const {
  if (_filemetas.find(name) == _filemetas.end()) {
    THROW(Aborted, lut::sprintf("file \"%s\" not exist in zip.", name));
  }

  return _filemetas.at(name);
}

int ZipFile::Impl::close() {
  if (!_fp) return 0;

  fclose(_fp);
  _fp = nullptr;

  return 0;
}

FILE *ZipFile::Impl::getFp() {
  return _fp;
}

// -----------------------------------------------------------------------------------------------+
// class ZipFile::FileReader                                                                      |
// -----------------------------------------------------------------------------------------------+

ZipFile::FileReader::FileReader()
    : _fp(nullptr),
      _pos(0) {
}
ZipFile::FileReader::~FileReader() {
}

void ZipFile::FileReader::init(
    std::shared_ptr<ZipFile::Impl> zipImpl,
    const std::string &filename) {
  _fileMeta = zipImpl->getFileMeta(filename);
  _fp = zipImpl->getFp();
}

int64_t ZipFile::FileReader::readData(Span<int8_t> buffer) {
  int64_t nRemain = _fileMeta.size - _pos;
  int64_t nRead = std::min(nRemain, static_cast<int64_t>(buffer.size()));

  // on EOF.
  if (nRead == 0) return 0;
  int ok = FSEEK64(_fp, _fileMeta.offset + _pos, SEEK_SET);
  if (ok != 0) THROW(Aborted, "failed to read zip file.");

  int n = fread(buffer.data(), nRead, 1, _fp);
  if (n != 1) THROW(Aborted, "failed to read zip file.");

  _pos += nRead;
  return nRead;
}

// -----------------------------------------------------------------------------------------------+
// class ZipFile                                                                                  |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<ZipFile> ZipFile::fromFile(const std::string &filename) {
  std::shared_ptr<ZipFile> zipFile{new ZipFile()};
  zipFile->_impl = std::make_shared<Impl>();
  zipFile->_impl->open(filename);

  return zipFile;
}

std::shared_ptr<Reader> ZipFile::open(const std::string &filename) const {
  std::shared_ptr<FileReader> reader = std::make_shared<FileReader>();
  reader->init(_impl, filename);

  return reader;
}

}  // namespace lut
