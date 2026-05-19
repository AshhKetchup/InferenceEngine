#pragma once

#include "gguf_model.h"
#include "tokenizer.h"
#include <cstdint>
#include <string>

class Config {
public:
  std::string arch;
  uint32_t vocab_size = 0;
  uint32_t context_length = 0;
  uint32_t embedding_length = 0;
  uint32_t block_count = 0;
  uint32_t feed_forward_length = 0;
  uint32_t attention_head_count = 0;
  uint32_t attention_head_count_kv = 0;
  uint32_t attention_key_length = 0;
  uint32_t attention_value_length = 0;
  float rms_norm_eps = 0.0f;
  float rope_freq_base = 0.0f;

  Config(GGUFModel *model, const Tokenizer &tok);

  uint32_t head_dim() const { return attention_key_length; }
  uint32_t kv_group_size() const {
    return attention_head_count_kv == 0
               ? 0
               : attention_head_count / attention_head_count_kv;
  }

  void print() const;
};
