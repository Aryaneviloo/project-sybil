
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sybil.config import SybilConfig
from src.sybil.models import SybilModelLoader
from src.sybil.baseline import generate_baseline_text
from src.sybil.speculative import SpeculativeEngine


def test_speculative_matches_baseline_exactly():
    config = SybilConfig(
        draft_model="gpt2",
        target_model="gpt2-medium",
        prompt="The most important concept in the field of Quantum mechanics is",
        max_new_tokens=20,
        num_speculative_tokens=5,
    )
    loader = SybilModelLoader(config)

    baseline_text = generate_baseline_text(
        loader.sovereign, loader.tokenizer, config.prompt,
        config.max_new_tokens, loader.device,
    )

    engine = SpeculativeEngine(
        loader.oracle, loader.sovereign, loader.tokenizer, loader.device,
        num_speculative_tokens=config.num_speculative_tokens,
    )
    speculative_text = engine.generate_text(config.prompt, config.max_new_tokens)

    print("Baseline:   ", repr(baseline_text))
    print("Speculative:", repr(speculative_text))
    assert baseline_text == speculative_text, "Outputs diverged — bug in accept/reject logic"
    print("PASSED: outputs are identical.")


if __name__ == "__main__":
    test_speculative_matches_baseline_exactly()