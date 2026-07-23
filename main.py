
import logging
import torch

from src.sybil.config import SybilConfig
from src.sybil.models import SybilModelLoader
from src.sybil.baseline import generate_baseline
from src.sybil.speculative import SpeculativeEngine
from src.sybil.timing import run_benchmark

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("sybil")

# Deliberately mixed: some narrow/technical (hypothesized higher acceptance),
# some open-ended/abstract (hypothesized lower acceptance).
PROMPTS = [
    ("technical", "The most important concept in the field of Quantum mechanics is"),
    ("technical", "In Python, a for loop is used to"),
    ("technical", "The mitochondria is the"),
    ("abstract",  "The fundamental nature of power is"),
    ("abstract",  "The meaning of a good life is"),
    ("abstract",  "What people misunderstand most about happiness is"),
]


def main():
    config = SybilConfig()
    torch.manual_seed(config.seed)
    loader = SybilModelLoader(config)

    engine = SpeculativeEngine(
        loader.oracle, loader.sovereign, loader.tokenizer, loader.device,
        num_speculative_tokens=config.num_speculative_tokens,
    )

    results = []

    for category, prompt in PROMPTS:
        def baseline_fn():
            return generate_baseline(
                loader.sovereign, loader.tokenizer, prompt,
                config.max_new_tokens, loader.device,
            )

        def speculative_fn():
            return engine.generate(prompt, config.max_new_tokens)

        baseline_result = run_benchmark(
            "baseline", baseline_fn, loader.device, config.num_trials, config.num_warmup
        )
        speculative_result = run_benchmark(
            "speculative", speculative_fn, loader.device, config.num_trials, config.num_warmup
        )
        _, avg_accept = engine.generate_with_stats(prompt, config.max_new_tokens)

        speedup = baseline_result.mean_time / speculative_result.mean_time
        results.append((category, prompt, avg_accept, speedup))

        logger.info("[%s] %r -> avg_accept=%.2f speedup=%.2fx",
                    category, prompt[:40], avg_accept, speedup)

    logger.info("=" * 70)
    for category, prompt, avg_accept, speedup in results:
        logger.info("%-10s avg_accept=%.2f  speedup=%.2fx  %r",
                     category, avg_accept, speedup, prompt[:50])

    technical_acc = [r[2] for r in results if r[0] == "technical"]
    abstract_acc = [r[2] for r in results if r[0] == "abstract"]
    logger.info("Mean acceptance — technical: %.2f, abstract: %.2f",
                sum(technical_acc) / len(technical_acc),
                sum(abstract_acc) / len(abstract_acc))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()