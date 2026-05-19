#include "config.h"
#include <cstdio>
#include <iostream>
using namespace std;

Config::Config(GGUFModel *model, const Tokenizer &tok) {
  gguf_ctx *ctx = model->getCtx();
  if (!ctx)
    return;

  // Pass 1: find architecture string.
  gguf_rewind(ctx);
  gguf_key key;
  while (gguf_get_key(ctx, &key)) {
    if (Callback::key_is(key, "general.architecture") &&
        key.type == GGUF_VALUE_TYPE_STRING) {
      arch.assign(key.val->string.string, key.val->string.len);
    }
    gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
  }

  if (arch.empty()) {
    fprintf(stderr, "Config: missing general.architecture key\n");
    return;
  }

  // Pass 2: read prefixed keys.
  string prefix = arch + ".";
  gguf_rewind(ctx);
  while (gguf_get_key(ctx, &key)) {
    // cout << key.name << " : " << key.type << endl;
    if (Callback::key_is(key, (prefix + "embedding_length").c_str()))
      embedding_length = key.val->uint32;
    else if (Callback::key_is(key, (prefix + "block_count").c_str()))
      block_count = key.val->uint32;
    else if (Callback::key_is(key, (prefix + "feed_forward_length").c_str())) {
      gguf_do_with_value(ctx, key.type, key.val, &feed_forward_length, 0, 0,
                         Callback::collect_ints);
      continue;
    } else if (Callback::key_is(key, (prefix + "context_length").c_str()))
      context_length = key.val->uint32;
    else if (Callback::key_is(key, (prefix + "attention.head_count").c_str()))
      attention_head_count = key.val->uint32;
    else if (Callback::key_is(key,
                              (prefix + "attention.head_count_kv").c_str()))
      attention_head_count_kv = key.val->uint32;
    else if (Callback::key_is(key, (prefix + "attention.key_length").c_str()))
      attention_key_length = key.val->uint32;
    else if (Callback::key_is(key, (prefix + "attention.value_length").c_str()))
      attention_value_length = key.val->uint32;
    else if (Callback::key_is(
                 key, (prefix + "attention.layer_norm_rms_epsilon").c_str()))
      rms_norm_eps = key.val->float32;
    else if (Callback::key_is(key, (prefix + "rope.freq_base").c_str()))
      rope_freq_base = key.val->float32;

    gguf_do_with_value(ctx, key.type, key.val, nullptr, 0, 0, nullptr);
  }

  vocab_size = static_cast<uint32_t>(tok.vocab_size());

  // Defaults.
  if (rope_freq_base == 0.0f)
    rope_freq_base = 10000.0f;
  if (rms_norm_eps == 0.0f)
    rms_norm_eps = 1e-6f;
  if (attention_key_length == 0 && attention_head_count > 0)
    attention_key_length = embedding_length / attention_head_count;
  if (attention_value_length == 0)
    attention_value_length = attention_key_length;
}

void Config::print() const {
  printf("arch                     : %s\n", arch.c_str());
  printf("vocab_size               : %u\n", vocab_size);
  printf("context_length           : %u\n", context_length);
  printf("embedding_length         : %u\n", embedding_length);
  printf("block_count              : %u\n", block_count);
  printf("attention.head_count     : %u\n", attention_head_count);
  printf("attention.head_count_kv  : %u\n", attention_head_count_kv);
  printf("attention.key_length     : %u\n", attention_key_length);
  printf("attention.value_length   : %u\n", attention_value_length);
  printf("rms_norm_eps             : %g\n", rms_norm_eps);
  printf("rope.freq_base           : %g\n", rope_freq_base);
  printf("feed_forward_length      : ");
  for (auto i : feed_forward_length)
    cout << i << " ";
  cout << endl;
}
