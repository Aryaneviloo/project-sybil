import torch
import time
import statistics
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class BenchmarkResult:
    name: str
    times_sec: List[float]
    tokens_generated: List[int]

    @property
    def mean_time(self) -> float:
        return statistics.mean(self.times_sec)

    @property
    def std_time(self) -> float:
        return statistics.stdev(self.times_sec) if len(self.times_sec) > 1 else 0.0

    @property
    def mean_tokens(self) -> float:
        return statistics.mean(self.tokens_generated)

    @property
    def tokens_per_sec(self) -> float:
        # Ratio of means, not mean-of-ratios: avoids being skewed by a single
        # trial with an unusually small denominator.
        return self.mean_tokens / self.mean_time

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.tokens_per_sec:.2f} tok/s "
            f"(mean_time={self.mean_time:.3f}s +/- {self.std_time:.3f}s, "
            f"n={len(self.times_sec)})"
        )


def sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_benchmark(name: str, fn: Callable[[], int], device: str,
                   num_trials: int, num_warmup: int) -> BenchmarkResult:
    """
    fn() should run one full generation and return the number of tokens
    generated (an int), so tests where a run exits early are still counted
    correctly.
    """
    for _ in range(num_warmup):
        sync_if_cuda(device)
        fn()
        sync_if_cuda(device)

    times, token_counts = [], []
    for _ in range(num_trials):
        sync_if_cuda(device)
        start = time.perf_counter()
        n_tokens = fn()
        sync_if_cuda(device)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        token_counts.append(n_tokens)

    return BenchmarkResult(name=name, times_sec=times, tokens_generated=token_counts)
