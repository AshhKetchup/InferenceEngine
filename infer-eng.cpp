#include "src/config.h"
#include "src/gguf_model.h"
#include "src/maths_op.h"
#include "src/tokenizer.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::flush;
using std::getline;
using std::string;
using std::vector;

static void repl(Tokenizer &tok) {
  string line;
  while (true) {
    cout << "> " << flush;
    if (!getline(std::cin, line))
      break;
    if (line == "bye")
      break;

    vector<int> raw = tok.encode(line, /*add_bos=*/true);
    cout << "raw  [" << raw.size() << "]:";
    for (int id : raw)
      cout << " " << id;
    cout << "\n";

    vector<int> chat = tok.format_chat_user(line);
    cout << "chat [" << chat.size() << "]:";
    for (int id : chat)
      cout << " " << id;
    cout << "\n";
    cout << "decoded: ";
    for (int id : chat)
      cout << tok.decode(id);
    cout << "\n";
  }
}

int main() {
  run_math_tests();

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

  Config cfg(&model, tok);
  cfg.print();

  repl(tok);

  return 0;
}
