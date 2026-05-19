#pragma once

extern "C" {
#include "../include/lib/gguflib.h"
}
#include <cstring>
#include <string>
#include <vector>

class GGUFModel {
  gguf_ctx *ctx;

public:
  explicit GGUFModel(const char *filename) { ctx = gguf_open(filename); }
  ~GGUFModel() {
    if (ctx)
      gguf_close(ctx);
  }
  gguf_ctx *getCtx() { return ctx; }
  bool ok() const { return ctx != nullptr; }
};

class Callback {
public:
  static void collect_strings(void *priv, uint32_t type, union gguf_value *val,
                              uint64_t /*in_array*/, uint64_t /*array_len*/) {
    if (type == GGUF_VALUE_TYPE_STRING) {
      auto *vec = static_cast<std::vector<std::string> *>(priv);
      vec->emplace_back(val->string.string, val->string.len);
    }
  }
  static void collect_floats(void *priv, uint32_t type, union gguf_value *val,
                             uint64_t /*in_array*/, uint64_t /*array_len*/) {
    if (type == GGUF_VALUE_TYPE_FLOAT32) {
      auto *vec = static_cast<std::vector<float> *>(priv);
      vec->push_back(val->float32);
    }
  }
  static void collect_ints(void *priv, uint32_t type, union gguf_value *val,
                           uint64_t /*in_array*/, uint64_t /*array_len*/) {
    if (type == GGUF_VALUE_TYPE_INT32) {
      auto *vec = static_cast<std::vector<int32_t> *>(priv);
      vec->push_back(val->int32);
    } else if (type == GGUF_VALUE_TYPE_UINT32) {
      auto *vec = static_cast<std::vector<uint32_t> *>(priv);
      vec->push_back(val->uint32);
    }
  }

  // GGUF key names are NOT null-terminated.
  static bool key_is(const gguf_key &key, const char *literal) {
    size_t lit_len = std::strlen(literal);
    return key.namelen == lit_len &&
           std::memcmp(key.name, literal, lit_len) == 0;
  }
};
