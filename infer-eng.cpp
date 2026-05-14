#include "gguflib.h"
#include<string>
using namespace std;


int main(){
  const char* filename = "./.models/gguf/gemma-4-E2B-it-UD-IQ2_M.gguf";
  gguf_ctx* ctx = gguf_open(filename);
  
}
