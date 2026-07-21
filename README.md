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
