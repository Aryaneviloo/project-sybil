import logging
import torch

from src.sybil.config import SybilConfig
from src.sybil.models import SybilModelLoader
from src.sybil.baseline import generate_baseline
from src.sybil.speculative import SpeculativeEngine
from src.sybil.timing import run_benchmark

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("sybil")


def main():
    config = SybilConfig()
    torch.manual_seed(config.seed)

    loader = SybilModelLoader(config)

    def baseline_fn():
        return generate_baseline(
            loader.sovereign, loader.tokenizer, config.prompt,
            config.max_new_tokens, loader.device,
        )

    baseline_result = run_benchmark(
        name="Baseline (KV-cached, target model only)",
        fn=baseline_fn,
        device=loader.device,
        num_trials=config.num_trials,
        num_warmup=config.num_warmup,
    )

    engine = SpeculativeEngine(
        loader.oracle, loader.sovereign, loader.tokenizer, loader.device,
        num_speculative_tokens=config.num_speculative_tokens,
    )

    def speculative_fn():
        return engine.generate(config.prompt, config.max_new_tokens)

    speculative_result = run_benchmark(
        name=f"Speculative (K={config.num_speculative_tokens})",
        fn=speculative_fn,
        device=loader.device,
        num_trials=config.num_trials,
        num_warmup=config.num_warmup,
    )

    logger.info("=" * 60)
    logger.info(baseline_result)
    logger.info(speculative_result)
    speedup = baseline_result.mean_time / speculative_result.mean_time
    logger.info("Speedup: %.2fx", speedup)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()