# InferenceEngine

A minimal LLM inference engine written in C++ from scratch. It loads models in
the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format,
parses their tokenizer and architecture metadata, and exposes a small REPL for
encoding/decoding text. Math primitives (dot product, matrix-vector multiply,
RMSNorm) are implemented by hand as the building blocks for a forward pass.

This is a learning-oriented project — the goal is to understand how modern
transformer inference works end-to-end without leaning on a framework. The
longer-term ambition is to shard a single model across multiple Macs over
**Thunderbolt**, so that a cluster of consumer machines can serve models far
larger than any one of them could hold in unified memory.

---

## Status at a glance

| Area                         | State        |
| ---------------------------- | ------------ |
| GGUF parsing (mmap)          | ✅ done      |
| Tokenizer (encode / decode)  | ✅ done      |
| Chat-template formatting     | ✅ done      |
| Architecture / `Config`      | ✅ done      |
| Math kernels (dot, matvec, rmsnorm) | ✅ done |
| Weight tensor loading        | 🚧 next     |
| Single-machine forward pass  | ⏳ planned  |
| KV cache + sampling          | ⏳ planned  |
| Thunderbolt-sharded inference | 🔭 vision  |

---

## What it does today

- Memory-maps a `.gguf` file and reads its metadata via
  [gguflib](https://github.com/antirez/gguf-tools) — zero-copy, no extra parser.
- Loads tokenizer vocab, scores, merges, and special tokens (BOS / EOS).
- Encodes / decodes text and formats chat-style prompts (`<start_of_turn>`…).
- Reads model architecture into a `Config`: vocab size, context length,
  embedding dim, block count, attention heads (Q / KV), RoPE base, head dim, etc.
- Provides hand-rolled math kernels: `dot_product`, `matvec`, `rmsnorm`,
  with a `run_math_tests()` self-check on startup.
- A REPL that tokenizes user input and prints raw + chat-formatted token IDs
  alongside the decoded string.

The transformer forward pass itself — embedding lookup, RoPE, attention,
FFN, residuals, output projection, sampling — is the next thing to build on
top of these primitives.

## Stack

- **C++17** for the engine, **C** for the bundled libs.
- **[gguflib](https://github.com/antirez/gguf-tools)** (`include/src/gguflib.c`)
  for parsing the GGUF container.
- **fp16** helpers (`include/src/fp16.c`) for half-precision conversions.
- `mmap` for zero-copy model loading.
- A plain `makefile` — no CMake, no external build system.

## Project layout

```
infer-eng.cpp       # entry point + REPL
src/
  config.{h,cpp}    # parses architecture metadata from GGUF
  tokenizer.{h,cpp} # vocab, encode/decode, chat formatting
  gguf_model.h      # thin wrapper over gguflib
  maths_op.{h,cpp}  # dot product, matvec, rmsnorm + tests
include/src/        # gguflib.c, fp16.c (vendored)
.models/gguf/       # put .gguf files here (gitignored)
```

## Setup

### Prerequisites

- A C++17 compiler (`g++` or `clang++`)
- `make`
- A GGUF model file

### Build

```sh
make
```

This produces an `infer-eng` binary in the project root.

### Get a model

The default path expected by `infer-eng.cpp` is:

```
./.models/gguf/gemma-4-E2B-it-UD-IQ2_M.gguf
```

Drop any GGUF model into `.models/gguf/` and update the `filename` in
`infer-eng.cpp` if it differs. Small quantized models (e.g. from Hugging Face,
Unsloth, or `bartowski`) work well for experimenting.

### Run

```sh
./infer-eng
```

You'll see the math self-tests, model metadata, then a `>` prompt. Type
anything to see how it tokenizes; type `bye` to exit.

```
> hello world
raw  [3]: 2 17534 2134
chat [9]: 2 105 ...
decoded: <bos><start_of_turn>user
hello world<end_of_turn>
```

### Clean

```sh
make clean
```

---

## Roadmap

### Near-term — finish single-machine inference

1. **Weight loading.** Walk GGUF tensors, dequantize the formats we care
   about (Q4_K, Q6_K, Q8_0, F16) into either fp32 working buffers or
   keep them packed and dequantize on the fly inside the kernels.
2. **Forward pass.** Embedding → N × {RMSNorm → GQA self-attention with
   RoPE → residual → RMSNorm → SwiGLU FFN → residual} → final RMSNorm →
   output projection. One block at a time, validated against a reference
   implementation (e.g. `llama.cpp`) on identical inputs.
3. **KV cache.** Per-layer key/value buffers sized by `Config.context_length`
   and `Config.kv_heads`. Append-on-decode, no recomputation.
4. **Sampling.** Greedy first, then top-k / top-p / temperature.
5. **Perf pass.** SIMD (NEON on Apple Silicon) for `matvec`, blocked
   quantized kernels, then Accelerate / AMX when it's worth it.

### Mid-term — quality of life

- Streaming token output in the REPL.
- A small HTTP / OpenAI-compatible endpoint so other tools can talk to it.
- Support for a few model families beyond Gemma (Llama 3, Qwen, Phi).
- Proper benchmarks: tokens/sec, time-to-first-token, memory residency.

### Long-term — Thunderbolt-sharded inference across multiple Macs

The headline goal: **run a model that doesn't fit in one Mac's unified
memory by splitting it across several Macs connected over Thunderbolt 4 / 5.**

#### Why Thunderbolt

A single Mac mini or MacBook tops out at the unified memory Apple ships it
with. Thunderbolt 4 gives ~40 Gbps and Thunderbolt 5 ~80–120 Gbps of bandwidth
between two machines, with sub-millisecond latency over IP-over-Thunderbolt
(`bridge0`) — an order of magnitude faster than typical home Wi-Fi / Ethernet
and competitive with budget datacenter interconnects. That's the right shape
of pipe for shipping hidden-state activations between transformer blocks.

#### Sharding strategy

The plan is **pipeline parallelism** as the first cut, since it's the
friendliest fit for transformer blocks and Thunderbolt's
point-to-point topology:

```
   ┌──────────┐  TB   ┌──────────┐  TB   ┌──────────┐
   │  Mac A   │──────▶│  Mac B   │──────▶│  Mac C   │
   │ blocks   │       │ blocks   │       │ blocks   │
   │  0..k    │       │ k+1..2k  │       │ 2k+1..N  │
   │ + embed  │       │          │       │ + lm_head│
   └──────────┘       └──────────┘       └──────────┘
        ▲                                     │
        └──────────── token out ◀─────────────┘
```

- Each node owns a contiguous slice of transformer blocks plus its share of
  the KV cache.
- Activations (a single hidden-state tensor of shape
  `[seq, d_model]` per token in decode) are forwarded down the pipe.
- For autoregressive decode this is **one tiny tensor per token per hop** —
  Thunderbolt is overkill for it bandwidth-wise, which is exactly what we
  want for low latency.
- Prefill is the heavy phase; we can micro-batch sequence chunks to keep all
  nodes busy instead of a single bubble-prone pass.

#### Possible later phase: tensor parallelism within a block

For very wide layers, splitting a single `matvec` across two Macs (column /
row split, all-reduce on the result) becomes attractive. Thunderbolt's
bandwidth makes this borderline-viable for fp16 / int8 matmuls; quantized
weights make the per-step transfer small enough to consider. This is a
"once the pipeline version actually works" problem.

#### Open questions to figure out

- **Transport.** Raw TCP over `bridge0` is the simplest start. RDMA-style
  zero-copy isn't on macOS, but `MSG_ZEROCOPY`-equivalents and tight
  ring-buffer designs can get close.
- **Discovery.** Static config first (a JSON listing each node's IP and
  block range), Bonjour / mDNS later.
- **Failure model.** A node dropping mid-decode kills the request; that's
  fine for v1. Checkpointed KV state is a v2 concern.
- **Heterogeneous nodes.** An M1 mini next to an M3 Max should still work —
  block assignment can be weighted by per-node throughput measured at
  startup.
- **Quantization split.** Each node only needs to hold its slice's weights,
  so a 70B model at Q4 (~40 GB) fits comfortably across three 16 GB Macs
  with headroom for KV cache.

#### Milestones

1. Working single-Mac decode (prerequisite — everything else is dead without it).
2. A "loopback" two-process version on one machine that talks over
   localhost sockets, with the same wire format Thunderbolt will use.
3. Two real Macs over a Thunderbolt cable, splitting blocks 50/50.
4. N-node ring with weighted block assignment and a tiny coordinator.
5. Benchmarks vs. running the same quantization on one machine — what's
   the latency cost per hop, and at what model size does the cluster
   actually win?

---

## Why build this

There are excellent inference engines already (`llama.cpp`, `mlx`, `vllm`).
This one exists to **learn the stack from the bottom up** and to chase a
specific question: *can a small pile of consumer Macs, wired together with
the cable already in the drawer, serve a model that none of them could
serve alone?* The single-machine engine is the price of admission; the
distributed runtime is the actual experiment.

Contributions, corrections, and "you're holding it wrong"-style feedback
are welcome.
