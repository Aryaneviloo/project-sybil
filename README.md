# Project Sybil

> An experimental speculative decoding engine built in PyTorch to explore faster Large Language Model inference through draft-and-verify generation.

---

## Overview

Large Language Models are often limited not by raw compute power, but by memory bandwidth. During autoregressive generation, the model repeatedly loads parameters and performs inference one token at a time, creating a bottleneck that prevents modern GPUs from reaching their full utilization.

Project Sybil investigates whether speculative decoding can reduce this inefficiency by allowing a lightweight draft model to generate multiple candidate tokens ahead of a larger target model.

The goal is not to build another chatbot, but to explore the systems and infrastructure techniques used to accelerate modern LLM serving.

---

## Architecture

Sybil implements a dual-model inference pipeline:

### Oracle (Draft Model)

A lightweight GPT-2 model responsible for rapidly proposing multiple future tokens.

* Model: GPT-2 (124M parameters)
* Purpose: Generate speculative continuations
* Optimized for speed

### Sovereign (Verifier Model)

A larger GPT-2 Medium model responsible for validating the Oracle's predictions.

* Model: GPT-2 Medium (355M parameters)
* Purpose: Verify speculative tokens
* Maintains output quality

### Generation Flow

Input Prompt
↓
Oracle drafts K tokens
↓
Sovereign verifies draft
↓
Accept valid tokens
↓
Reject invalid branch
↓
Continue generation

---

## Speculative Decoding Strategy

For each iteration:

1. The Oracle proposes K future tokens.
2. The Sovereign evaluates the drafted sequence.
3. Accepted tokens are committed to the output.
4. On divergence, invalid tokens are discarded.
5. The Sovereign's prediction becomes the new source of truth.

This process attempts to reduce the number of expensive target-model generation steps while preserving output fidelity.

---

## Current Results

Hardware:

* NVIDIA GTX 1650
* CUDA acceleration enabled

Benchmark:

| Method                             | Throughput       |
| ---------------------------------- | ---------------- |
| Standard Autoregressive Generation | 3.64 tokens/sec  |
| Sybil Prototype                    | 37.51 tokens/sec |

Observed draft-generation speedup:

**~10.3× increase in throughput**

---

## Current Limitations

This project is an experimental prototype.

While throughput improvements were observed, output quality degradation was also detected during testing.

Potential causes under investigation include:

* Acceptance policy design
* Draft/verifier distribution mismatch
* Verification logic edge cases
* KV-cache synchronization issues

As a result, current benchmark numbers should be treated as exploratory rather than production-ready results.

---

## Future Work

### Core Engine

* Acceptance-rate telemetry
* Dynamic speculative window sizing
* Better draft model selection
* Adaptive verification strategies

### Performance

* KV-cache optimization
* Batched verification
* Memory profiling
* CUDA kernel analysis

### Evaluation

* Acceptance-rate metrics
* Quality benchmarking
* Latency breakdowns
* Throughput vs quality tradeoff analysis

---

## Why This Project Exists

Most educational LLM projects focus on prompt engineering or API integrations.

Project Sybil focuses on inference infrastructure.

The objective is to understand the systems-level challenges behind modern LLM serving, including:

* Transformer inference
* GPU utilization
* Memory bandwidth constraints
* Speculative decoding
* Verification pipelines
* Performance engineering

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* CUDA
* GPT-2
* GPT-2 Medium

---

## Running the Project

```bash
git clone https://github.com/yourusername/project-sybil.git

cd project-sybil

pip install -r requirements.txt

python main.py
```

---

## Disclaimer

Project Sybil is an experimental research project intended for learning and exploration of speculative decoding techniques. The implementation is actively evolving and should not be considered a production inference engine.
# Project Sybil

An experimental speculative decoding engine, built from scratch to actually
measure whether the technique pays off at small/medium model scales — not
just implement it and assume it does.

## What speculative decoding is

LLM decoding is memory-bandwidth-bound: generating one token means loading
the entire model's weights from memory to do a tiny amount of actual math.
The GPU spends most of its time waiting on memory, not computing.

Speculative decoding exploits an asymmetry: verifying K candidate tokens
against a model in one batched forward pass costs almost the same as
verifying 1, because the dominant cost (loading weights) is paid once
either way. So a cheap "draft" model guesses several tokens ahead, and the
expensive "target" model checks all of them in a single pass — if the
guesses are good, you get several tokens for close to the price of one
expensive step.

Two things decide whether this actually wins in practice:
- **Acceptance rate** — how often the draft's guesses match what the
  target would have produced anyway.
- **Absolute model scale** — how much of each forward pass's cost is
  "loading weights" (which batches for free) vs. fixed per-call overhead
  (Python dispatch, kernel launch, CUDA sync — which does NOT batch away).

## What's actually implemented

- KV-cached baseline generation (O(n), not the O(n²) naive re-feed)
- KV-cached speculative engine (Oracle/Sovereign), with correct cache
  trimming when a draft token is rejected
- Rigorous benchmarking (CUDA sync, warmup exclusion, multi-trial
  averaging) — no single-run numbers anywhere in this project
- An equivalence test proving the speculative engine's output is
  byte-for-byte identical to the baseline's, for greedy decoding —
  this is a mathematical property, not an approximation, and the test
  actually caught a real bug (see below) before it shipped
- A multi-prompt benchmark sweep, comparing acceptance/speedup across
  different prompt styles and model pairs

## Findings so far

**A real correctness bug was found and fixed via the equivalence test.**
An early version had an off-by-one in which logits predict which token,
plus stale cache entries surviving a rejected draft — it ran without
erroring and produced plausible-looking text, but the output was wrong.
All early speed/acceptance numbers from before this fix are invalid and
were discarded; every number below is post-fix.

**Speculative decoding does not pay off at GPT-2 / small-Qwen scale, even
with good acceptance rates.** Tested pairs:

| Draft | Target | Acceptance (tokens/round, K=5) | Speedup |
|---|---|---|---|
| GPT-2 (124M) | GPT-2 Medium (355M) | 2.6–4.2 (~52–84%) | 0.49–0.66x |
| GPT-2 (124M) | GPT-2 Large (774M) | 2.8–4.2 (~56–84%) | 0.57–1.02x |
| Qwen2.5-0.5B | Qwen2.5-1.5B | 3.1–4.2 (~62–84%) | 0.41–0.53x |

Acceptance is consistently decent-to-good across all three pairs — the
draft models are guessing right well over half the time. Speedup is
consistently *below* 1.0x anyway. This means the bottleneck isn't draft
quality — it's that these models are still small enough that fixed
per-call overhead (not memory bandwidth) dominates each forward pass, so
"batched verification is nearly free" doesn't hold yet at this scale.

Moving from GPT-2 to same-family, well-tokenizer-aligned Qwen2.5 models
didn't help either — alignment matters for acceptance rate, but scale is
still the binding constraint for actual speedup.

**A "prompt specificity affects acceptance" hypothesis, formed from 2
early examples, did not survive a 6-prompt sweep** — the technical/abstract
acceptance gap flipped direction between model pairs, indicating noise
rather than a real effect. Retired.

## Hardware

Tested on a laptop GTX 1650 (8GB VRAM). Cross-hardware comparison against
a CPU-only (Intel 1205u, 8GB RAM) machine is planned but not yet run.

## What's next

- Test a genuinely large target model (7B-class, likely via 4-bit
  quantization to fit 8GB VRAM) to check whether speedup ever crosses
  1.0x at a larger scale — this is the open question the project hasn't
  answered yet
- Cross-hardware comparison (GPU vs. CPU-only)
- Sampling-based generation (temperature > 0) with proper probabilistic
  rejection-sampling acceptance, instead of the current exact-match rule
  (which is provably correct for greedy but doesn't generalize to sampling)

## Running it

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

Runs the multi-prompt benchmark sweep defined in `main.py`, printing
per-prompt acceptance rate and speedup, plus category averages.