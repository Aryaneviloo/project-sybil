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

A lightweight model responsible for rapidly proposing multiple future tokens.

### Sovereign (Verifier Model)

A larger model responsible for validating the Oracle's predictions. Maintains output quality.

### Generation Flow

Input Prompt → Oracle drafts K tokens → Sovereign verifies draft → Accept valid tokens → Reject invalid branch → Continue generation

---

## Speculative Decoding Strategy

For each iteration:

1. The Oracle proposes K future tokens.
2. The Sovereign evaluates the drafted sequence in a single batched forward pass.
3. Accepted tokens are committed to the output.
4. On divergence, invalid tokens are discarded and the Sovereign's correction token is used.
5. The KV cache is trimmed to drop rejected-token entries before the next round.

---

## Current Results

Hardware: NVIDIA GTX 1650 (8GB VRAM)

These are post-fix results. An early version had an off-by-one in logit indexing plus stale cache entries surviving a rejected draft; it ran without erroring and produced plausible-looking but wrong output. All pre-fix numbers were discarded.

Tested draft/target pairs (greedy decoding, K = 5):

| Draft | Target | Acceptance (tokens/round, K=5) | Speedup |
|---|---|---|---|
| GPT-2 (124M) | GPT-2 Medium (355M) | 2.6–4.2 (~52–84%) | 0.49–0.66× |
| GPT-2 (124M) | GPT-2 Large (774M) | 2.8–4.2 (~56–84%) | 0.57–1.02× |
| Qwen2.5-0.5B | Qwen2.5-1.5B | 3.1–4.2 (~62–84%) | 0.41–0.53× |

Acceptance is consistently decent-to-good across all three pairs — the draft models guess right well over half the time. Speedup is consistently **below 1.0×** anyway. This means the bottleneck isn't draft quality — it's that these models are still small enough that fixed per-call overhead (Python dispatch, kernel launch, CUDA sync — which does NOT batch away) dominates each forward pass, so "batched verification is nearly free" doesn't hold yet at this scale.

A "prompt specificity affects acceptance" hypothesis, formed from early examples, did not survive a 6-prompt sweep — the technical/abstract acceptance gap flipped direction between model pairs, indicating noise rather than a real effect. Retired.

---

## Correctness

`tests/test_equivalence.py` proves the speculative engine's output is byte-for-byte identical to the baseline's for greedy decoding — a mathematical property, not an approximation. This test caught the real off-by-one / stale-cache bug described above before it shipped.

---

## Current Limitations

This project is an experimental prototype.

While acceptance rates are reasonable, observed speedup is below 1.0× at small-model scale. Output quality degradation was also detected during early testing.

Potential causes under investigation include:

* Acceptance policy design
* Draft/verifier distribution mismatch
* Verification logic edge cases
* KV-cache synchronization issues

As a result, current benchmark numbers should be treated as exploratory rather than production-ready results.

---

## Default Configuration

`src/sybil/config.py` defaults to:

* Draft (Oracle): `Qwen/Qwen2.5-0.5B`
* Target (Sovereign): `Qwen/Qwen2.5-1.5B`
* Speculative window (K): 5 tokens
* Greedy decoding (`do_sample = False`)

The equivalence test and results table above also cover GPT-2 / GPT-2 Medium and GPT-2 / GPT-2 Large.

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

### Open Questions

* Test a genuinely large target model (7B-class, likely via 4-bit quantization to fit 8GB VRAM) to check whether speedup ever crosses 1.0× at a larger scale — this is the open question the project hasn't answered yet.
* Cross-hardware comparison (GPU vs. CPU-only).
* Sampling-based generation (temperature > 0) with proper probabilistic rejection-sampling acceptance, instead of the current exact-match rule (provably correct for greedy but not for sampling).

## Running it

```bash
cd /home/evilooo/project-sybil/project-sybil
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

Runs the multi-prompt benchmark sweep defined in `main.py`, printing per-prompt acceptance rate and speedup, plus category averages.

---

## License

MIT License. See `LICENSE`.