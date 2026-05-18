extern "C" {
#include "./include/lib/gguflib.h"
}
#include <cstdio>
#include <cstring>
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
};

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

  return 0;
}
