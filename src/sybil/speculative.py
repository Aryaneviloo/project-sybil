import torch
from dataclasses import dataclass
from typing import Any
import logging
logger = logging.getLogger("sybil")


def trim_cache(past_key_values, keep_length: int):

    """Return a cache truncated to the first `keep_length` positions."""
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_length)
        return past_key_values

    trimmed = tuple(
        (key[:, :, :keep_length, :], value[:, :, :keep_length, :])
        for key, value in past_key_values
    )
    return trimmed


@dataclass
class SpeculativeEngine:
    oracle: Any
    sovereign: Any
    tokenizer: Any
    device: str
    num_speculative_tokens: int = 5

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> int:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        oracle_out = self.oracle(input_ids, use_cache=True)
        sovereign_out = self.sovereign(input_ids, use_cache=True)
        oracle_cache = oracle_out.past_key_values
        sovereign_cache = sovereign_out.past_key_values

        last_token = input_ids[:, -1:]
        accepted_len = prompt_len
        n_generated = 0
        num_rounds = 0

        while n_generated < max_new_tokens:
            num_rounds += 1
            draft_tokens, oracle_cache = self._draft(last_token, oracle_cache)

            accepted_tokens, sovereign_cache, new_last_token = self._verify(
                draft_tokens, sovereign_cache, accepted_len
            )

            num_accepted = len(accepted_tokens)
            accepted_len += num_accepted

            oracle_cache = trim_cache(oracle_cache, accepted_len)
            sovereign_cache = trim_cache(sovereign_cache, accepted_len)

            n_generated += num_accepted
            last_token = new_last_token

            if n_generated >= max_new_tokens:
                break

        
        avg_accept = n_generated / num_rounds if num_rounds else 0.0
        logger.info(
            "avg accepted tokens/round: %.2f out of K=%d (%d rounds)",
            avg_accept, self.num_speculative_tokens, num_rounds,
        )
        return n_generated

    def _draft(self, last_token, oracle_cache):
        draft_tokens = []
        current_token = last_token
        cache = oracle_cache

        for _ in range(self.num_speculative_tokens):
            out = self.oracle(current_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_token)
            current_token = next_token

        return draft_tokens, cache

    def _verify(self, draft_tokens, sovereign_cache, accepted_len):
        draft_ids = torch.cat(draft_tokens, dim=1)
        out = self.sovereign(draft_ids, past_key_values=sovereign_cache, use_cache=True)
        cache = out.past_key_values

        accepted_tokens = []
        for i in range(self.num_speculative_tokens):
            target_token = torch.argmax(out.logits[:, i, :], dim=-1, keepdim=True)
            draft_token = draft_tokens[i]

            if draft_token.item() == target_token.item():
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(target_token)
                break

        last_token = accepted_tokens[-1]
        return accepted_tokens, cache, last_token