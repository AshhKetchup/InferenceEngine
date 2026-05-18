extern "C" {
#include "./include/lib/gguflib.h"
}
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

class GGUFModel {
private:
  gguf_ctx *ctx;

public:
  GGUFModel(const char *filename) { ctx = gguf_open(filename); }
  ~GGUFModel() {
    if (ctx)
      gguf_close(ctx);
  }
  gguf_ctx *getCtx() { return ctx; }
  bool ok() const { return ctx != nullptr; }
};

class Tokenizer {
  vector<string> tokens;
  vector<float> scores;
  vector<int32_t> token_type;
  vector<string> merge_rules;
  unordered_map<string, int> token2id;
  uint32_t bos_id = 0;
  uint32_t eos_id = 0;
  string model_type;

  // ---------- callbacks for gguf_do_with_value ----------
  static void collect_strings(void *priv, uint32_t type, union gguf_value *val,
                              uint64_t in_array, uint64_t array_len) {
    if (type == GGUF_VALUE_TYPE_STRING) {
      auto *vec = static_cast<vector<string> *>(priv);
      vec->emplace_back(val->string.string, val->string.len);
    }
  }
  static void collect_floats(void *priv, uint32_t type, union gguf_value *val,
                             uint64_t in_array, uint64_t array_len) {
    if (type == GGUF_VALUE_TYPE_FLOAT32) {
      auto *vec = static_cast<vector<float> *>(priv);
      vec->push_back(val->float32);
    }
  }
  static void collect_ints(void *priv, uint32_t type, union gguf_value *val,
                           uint64_t in_array, uint64_t array_len) {
    if (type == GGUF_VALUE_TYPE_INT32) {
      auto *vec = static_cast<vector<int32_t> *>(priv);
      vec->push_back(val->int32);
    }
  }

  // Compare a (non-null-terminated) GGUF key name against a literal.
  static bool key_is(const gguf_key &key, const char *literal) {
    size_t lit_len = strlen(literal);
    return key.namelen == lit_len && memcmp(key.name, literal, lit_len) == 0;
  }

public:
  Tokenizer(GGUFModel *model) {
    gguf_ctx *ctx = model->getCtx();
    if (!ctx)
      return;

    gguf_rewind(ctx);
    gguf_key key;

    while (gguf_get_key(ctx, &key)) {
      if (key_is(key, "tokenizer.ggml.model") &&
          key.type == GGUF_VALUE_TYPE_STRING) {
        model_type.assign(key.val->string.string, key.val->string.len);
        gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
      } else if (key_is(key, "tokenizer.ggml.tokens")) {
        gguf_do_with_value(ctx, key.type, key.val, &tokens, 0, 0,
                           collect_strings);
      } else if (key_is(key, "tokenizer.ggml.scores")) {
        gguf_do_with_value(ctx, key.type, key.val, &scores, 0, 0,
                           collect_floats);
      } else if (key_is(key, "tokenizer.ggml.token_type")) {
        gguf_do_with_value(ctx, key.type, key.val, &token_type, 0, 0,
                           collect_ints);
      } else if (key_is(key, "tokenizer.ggml.merges")) {
        gguf_do_with_value(ctx, key.type, key.val, &merge_rules, 0, 0,
                           collect_strings);
      } else if (key_is(key, "tokenizer.ggml.bos_token_id") &&
                 key.type == GGUF_VALUE_TYPE_UINT32) {
        bos_id = key.val->uint32;
        gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
      } else if (key_is(key, "tokenizer.ggml.eos_token_id") &&
                 key.type == GGUF_VALUE_TYPE_UINT32) {
        eos_id = key.val->uint32;
        gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
      } else {
        // Skip every other key — must consume the value to advance ctx->off.
        gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
      }
    }

    // Build reverse lookup: string -> id
    token2id.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
      token2id.emplace(tokens[i], static_cast<int>(i));
    }
  }

  size_t vocab_size() const { return tokens.size(); }
  const string &decode(int id) const { return tokens.at(id); }
  int encode_token(const string &s) const {
    auto it = token2id.find(s);
    return it == token2id.end() ? -1 : it->second;
  }
  const string &type() const { return model_type; }
  uint32_t bos() const { return bos_id; }
  uint32_t eos() const { return eos_id; }
  size_t merges_count() const { return merge_rules.size(); }
  size_t scores_count() const { return scores.size(); }
  // Byte-fallback: SentencePiece reserves 256 tokens "<0x00>" .. "<0xFF>"
  // for any byte that can't be matched as a piece. We resolve them lazily
  // on first encode and cache.
  mutable vector<int> byte_fallback;
  void ensure_byte_fallback() const {
    if (!byte_fallback.empty()) return;
    auto &bf = const_cast<vector<int> &>(byte_fallback);
    bf.assign(256, -1);
    char buf[8];
    for (int b = 0; b < 256; ++b) {
      snprintf(buf, sizeof(buf), "<0x%02X>", b);
      auto it = token2id.find(buf);
      if (it != token2id.end()) bf[b] = it->second;
    }
  }

  // SentencePiece BPE encode. Returns token IDs.
  // Algorithm:
  //   1. Replace spaces with U+2581 ("▁"), prepend one ▁.
  //   2. Split into UTF-8 codepoint pieces. If a codepoint isn't in the
  //      vocab, emit per-byte fallback tokens "<0xNN>" instead.
  //   3. Repeatedly merge the adjacent pair whose concatenation has the
  //      highest score in the vocab, until no more merges apply.
  //   4. Optionally prepend BOS.
  vector<int> encode(const string &text, bool add_bos = false) const {
    static const string SPIECE = "\xE2\x96\x81"; // ▁ U+2581
    ensure_byte_fallback();

    // Step 1: prefix ▁ and replace spaces with ▁.
    string s = SPIECE;
    for (char c : text) {
      if (c == ' ') s += SPIECE;
      else s += c;
    }

    // Step 2: split into UTF-8 codepoints. Fall back to per-byte tokens
    // for codepoints missing from the vocab.
    vector<string> pieces;
    vector<int> ids;
    for (size_t i = 0; i < s.size();) {
      unsigned char b = static_cast<unsigned char>(s[i]);
      size_t len = 1;
      if ((b & 0x80) == 0)        len = 1;
      else if ((b & 0xE0) == 0xC0) len = 2;
      else if ((b & 0xF0) == 0xE0) len = 3;
      else if ((b & 0xF8) == 0xF0) len = 4;
      string piece = s.substr(i, len);
      auto it = token2id.find(piece);
      if (it != token2id.end()) {
        pieces.push_back(piece);
        ids.push_back(it->second);
      } else {
        // Byte fallback: one token per raw byte.
        for (size_t k = 0; k < len; ++k) {
          unsigned char rb = static_cast<unsigned char>(s[i + k]);
          int bid = byte_fallback[rb];
          if (bid < 0) return {}; // vocab has no byte fallback at all
          pieces.push_back(string(1, s[i + k]));
          ids.push_back(bid);
        }
      }
      i += len;
    }

    // Step 3: greedy highest-score merge until no pair is mergeable.
    while (pieces.size() >= 2) {
      float best_score = -1e30f;
      int best_idx = -1;
      int best_id = -1;
      string best_merged;
      for (size_t i = 0; i + 1 < pieces.size(); ++i) {
        string merged = pieces[i] + pieces[i + 1];
        auto it = token2id.find(merged);
        if (it == token2id.end()) continue;
        float sc = scores[it->second];
        if (sc > best_score) {
          best_score = sc;
          best_idx = static_cast<int>(i);
          best_id = it->second;
          best_merged = std::move(merged);
        }
      }
      if (best_idx < 0) break;
      pieces[best_idx] = best_merged;
      ids[best_idx] = best_id;
      pieces.erase(pieces.begin() + best_idx + 1);
      ids.erase(ids.begin() + best_idx + 1);
    }

    if (add_bos) ids.insert(ids.begin(), static_cast<int>(bos_id));
    return ids;
  }

  // Gemma-IT chat template. Wraps a single user turn:
  //   <bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
  // Special tokens are looked up by literal string so we don't hardcode IDs.
  vector<int> format_chat_user(const string &user_msg) const {
    vector<int> out;
    auto push_special = [&](const char *lit) {
      auto it = token2id.find(lit);
      if (it != token2id.end()) out.push_back(it->second);
    };
    out.push_back(static_cast<int>(bos_id));
    push_special("<start_of_turn>");
    // "user\n" + message body — encode WITHOUT bos, no leading ▁ space confusion:
    // SentencePiece will still add a ▁ prefix; that's expected for Gemma.
    vector<int> head = encode("user\n" + user_msg, /*add_bos=*/false);
    out.insert(out.end(), head.begin(), head.end());
    push_special("<end_of_turn>");
    // newline between turns
    vector<int> nl = encode("\n", /*add_bos=*/false);
    out.insert(out.end(), nl.begin(), nl.end());
    push_special("<start_of_turn>");
    vector<int> tail = encode("model\n", /*add_bos=*/false);
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
  }
};

static void repl(Tokenizer &tok) {
  string line;
  while (true) {
    cout << "> " << flush;
    if (!getline(cin, line))
      break;
    if (line == "bye")
      break;

    vector<int> raw = tok.encode(line, /*add_bos=*/true);
    cout << "raw  [" << raw.size() << "]:";
    for (int id : raw) cout << " " << id;
    cout << "\n";

    vector<int> chat = tok.format_chat_user(line);
    cout << "chat [" << chat.size() << "]:";
    for (int id : chat) cout << " " << id;
    cout << "\n";
    cout << "decoded: ";
    for (int id : chat) cout << tok.decode(id);
    cout << "\n";
  }
}

int main() {
  const char *filename = "./.models/gguf/gemma-4-E2B-it-UD-IQ2_M.gguf";

  GGUFModel model(filename);
  if (!model.ok()) {
    fprintf(stderr, "Failed to open GGUF file: %s\n", filename);
    return 1;
  }

  Tokenizer tok(&model);

  printf("tokenizer model : %s\n", tok.type().c_str());
  printf("vocab size      : %zu\n", tok.vocab_size());
  printf("scores          : %zu\n", tok.scores_count());
  printf("merges          : %zu\n", tok.merges_count());
  printf("bos_token_id    : %u\n", tok.bos());
  printf("eos_token_id    : %u\n", tok.eos());

  if (tok.vocab_size() > 0) {
    printf("token[0]        : \"%s\"\n", tok.decode(0).c_str());
    printf("token[%u] (bos) : \"%s\"\n", tok.bos(),
           tok.decode(tok.bos()).c_str());
    printf("token[%u] (eos) : \"%s\"\n", tok.eos(),
           tok.decode(tok.eos()).c_str());
  }

  repl(tok);

  return 0;
}
