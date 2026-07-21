import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sybil.timing import run_benchmark, BenchmarkResult
from src.sybil.config import SybilConfig


def test_benchmark_result_tokens_per_sec():
    result = BenchmarkResult(name="test", times_sec=[1.0, 2.0], tokens_generated=[10, 20])
    # mean_time = 1.5, mean_tokens = 15 -> 10 tok/s
    assert abs(result.tokens_per_sec - 10.0) < 1e-9


def test_benchmark_result_std_with_one_trial():
    result = BenchmarkResult(name="test", times_sec=[1.0], tokens_generated=[10])
    assert result.std_time == 0.0


def test_run_benchmark_excludes_warmup_from_timed_trials():
    calls = []

    def fn():
        calls.append(1)
        return 5

    result = run_benchmark("test", fn, device="cpu", num_trials=3, num_warmup=2)
    assert len(calls) == 5          # 2 warmup + 3 timed
    assert len(result.times_sec) == 3
    assert len(result.tokens_generated) == 3
    assert all(t == 5 for t in result.tokens_generated)


def test_config_defaults_are_sane():
    config = SybilConfig()
    assert config.num_warmup >= 1
    assert config.num_trials >= 2  # need >=2 for a meaningful std
    assert config.resolved_device() in ("cpu", "cuda")


if __name__ == "__main__":
    test_benchmark_result_tokens_per_sec()
    test_benchmark_result_std_with_one_trial()
    test_run_benchmark_excludes_warmup_from_timed_trials()
    test_config_defaults_are_sane()
    print("All tests passed.")
