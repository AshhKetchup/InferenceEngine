#include "tokenizer.h"
#include <cstdio>
#include <utility>

using std::string;
using std::vector;

Tokenizer::Tokenizer(GGUFModel *model) {
  gguf_ctx *ctx = model->getCtx();
  if (!ctx)
    return;

  gguf_rewind(ctx);
  gguf_key key;

  while (gguf_get_key(ctx, &key)) {
    if (Callback::key_is(key, "tokenizer.ggml.model") &&
        key.type == GGUF_VALUE_TYPE_STRING) {
      model_type.assign(key.val->string.string, key.val->string.len);
      gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
    } else if (Callback::key_is(key, "tokenizer.ggml.tokens")) {
      gguf_do_with_value(ctx, key.type, key.val, &tokens, 0, 0,
                         Callback::collect_strings);
    } else if (Callback::key_is(key, "tokenizer.ggml.scores")) {
      gguf_do_with_value(ctx, key.type, key.val, &scores, 0, 0,
                         Callback::collect_floats);
    } else if (Callback::key_is(key, "tokenizer.ggml.token_type")) {
      gguf_do_with_value(ctx, key.type, key.val, &token_type, 0, 0,
                         Callback::collect_ints);
    } else if (Callback::key_is(key, "tokenizer.ggml.merges")) {
      gguf_do_with_value(ctx, key.type, key.val, &merge_rules, 0, 0,
                         Callback::collect_strings);
    } else if (Callback::key_is(key, "tokenizer.ggml.bos_token_id") &&
               key.type == GGUF_VALUE_TYPE_UINT32) {
      bos_id = key.val->uint32;
      gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
    } else if (Callback::key_is(key, "tokenizer.ggml.eos_token_id") &&
               key.type == GGUF_VALUE_TYPE_UINT32) {
      eos_id = key.val->uint32;
      gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
    } else {
      gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
    }
  }

  token2id.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    token2id.emplace(tokens[i], static_cast<int>(i));
  }
}

int Tokenizer::encode_token(const string &s) const {
  auto it = token2id.find(s);
  return it == token2id.end() ? -1 : it->second;
}

void Tokenizer::ensure_byte_fallback() const {
  if (!byte_fallback.empty())
    return;
  auto &bf = const_cast<vector<int> &>(byte_fallback);
  bf.assign(256, -1);
  char buf[8];
  for (int b = 0; b < 256; ++b) {
    snprintf(buf, sizeof(buf), "<0x%02X>", b);
    auto it = token2id.find(buf);
    if (it != token2id.end())
      bf[b] = it->second;
  }
}

vector<int> Tokenizer::encode(const string &text, bool add_bos) const {
  static const string SPIECE = "\xE2\x96\x81"; // ▁ U+2581
  ensure_byte_fallback();

  string s = SPIECE;
  for (char c : text) {
    if (c == ' ')
      s += SPIECE;
    else
      s += c;
  }

  vector<string> pieces;
  vector<int> ids;
  for (size_t i = 0; i < s.size();) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((b & 0x80) == 0)
      len = 1;
    else if ((b & 0xE0) == 0xC0)
      len = 2;
    else if ((b & 0xF0) == 0xE0)
      len = 3;
    else if ((b & 0xF8) == 0xF0)
      len = 4;
    string piece = s.substr(i, len);
    auto it = token2id.find(piece);
    if (it != token2id.end()) {
      pieces.push_back(piece);
      ids.push_back(it->second);
    } else {
      for (size_t k = 0; k < len; ++k) {
        unsigned char rb = static_cast<unsigned char>(s[i + k]);
        int bid = byte_fallback[rb];
        if (bid < 0)
          return {};
        pieces.push_back(string(1, s[i + k]));
        ids.push_back(bid);
      }
    }
    i += len;
  }

  while (pieces.size() >= 2) {
    float best_score = -1e30f;
    int best_idx = -1;
    int best_id = -1;
    string best_merged;
    for (size_t i = 0; i + 1 < pieces.size(); ++i) {
      string merged = pieces[i] + pieces[i + 1];
      auto it = token2id.find(merged);
      if (it == token2id.end())
        continue;
      float sc = scores[it->second];
      if (sc > best_score) {
        best_score = sc;
        best_idx = static_cast<int>(i);
        best_id = it->second;
        best_merged = std::move(merged);
      }
    }
    if (best_idx < 0)
      break;
    pieces[best_idx] = best_merged;
    ids[best_idx] = best_id;
    pieces.erase(pieces.begin() + best_idx + 1);
    ids.erase(ids.begin() + best_idx + 1);
  }

  if (add_bos)
    ids.insert(ids.begin(), static_cast<int>(bos_id));
  return ids;
}

vector<int> Tokenizer::format_chat_user(const string &user_msg) const {
  vector<int> out;
  auto push_special = [&](const char *lit) {
    auto it = token2id.find(lit);
    if (it != token2id.end())
      out.push_back(it->second);
  };
  out.push_back(static_cast<int>(bos_id));
  push_special("<start_of_turn>");
  vector<int> head = encode("user\n" + user_msg, /*add_bos=*/false);
  out.insert(out.end(), head.begin(), head.end());
  push_special("<end_of_turn>");
  vector<int> nl = encode("\n", /*add_bos=*/false);
  out.insert(out.end(), nl.begin(), nl.end());
  push_special("<start_of_turn>");
  vector<int> tail = encode("model\n", /*add_bos=*/false);
  out.insert(out.end(), tail.begin(), tail.end());
  return out;
}
