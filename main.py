
import logging
import torch

from src.sybil.config import SybilConfig
from src.sybil.models import SybilModelLoader
from src.sybil.baseline import generate_baseline
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

    result = run_benchmark(
        name="Baseline (KV-cached, target model only)",
        fn=baseline_fn,
        device=loader.device,
        num_trials=config.num_trials,
        num_warmup=config.num_warmup,
    )

    logger.info("=" * 60)
    logger.info(result)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
