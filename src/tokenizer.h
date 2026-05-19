#pragma once

#include "gguf_model.h"
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
  std::vector<std::string> tokens;
  std::vector<float> scores;
  std::vector<int32_t> token_type;
  std::vector<std::string> merge_rules;
  std::unordered_map<std::string, int> token2id;
  uint32_t bos_id = 0;
  uint32_t eos_id = 0;
  std::string model_type;

  // Byte-fallback cache, lazily populated.
  mutable std::vector<int> byte_fallback;
  void ensure_byte_fallback() const;

public:
  explicit Tokenizer(GGUFModel *model);

  size_t vocab_size() const { return tokens.size(); }
  const std::string &decode(int id) const { return tokens.at(id); }
  int encode_token(const std::string &s) const;
  const std::string &type() const { return model_type; }
  uint32_t bos() const { return bos_id; }
  uint32_t eos() const { return eos_id; }
  size_t merges_count() const { return merge_rules.size(); }
  size_t scores_count() const { return scores.size(); }

  std::vector<int> encode(const std::string &text, bool add_bos = false) const;
  std::vector<int> format_chat_user(const std::string &user_msg) const;
};
