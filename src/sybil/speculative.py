
from dataclasses import dataclass
from typing import Any
import torch


def trim_cache(past_key_values, keep_length: int):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_length)
        return past_key_values
    return tuple(
        (key[:, :, :keep_length, :], value[:, :, :keep_length, :])
        for key, value in past_key_values
    )


@dataclass
class SpeculativeEngine:
    oracle: Any
    sovereign: Any
    tokenizer: Any
    device: str
    num_speculative_tokens: int = 5

    def _draft(self, oracle_pending, oracle_cache):
        """
        First draft token comes free from oracle_pending (already computed
        by whoever extended the cache last). Each subsequent token costs
        one real forward call.
        """
        K = self.num_speculative_tokens
        draft_tokens = []
        current_pending = oracle_pending
        cache = oracle_cache

        for j in range(K):
            token = torch.argmax(current_pending, dim=-1, keepdim=True)
            draft_tokens.append(token)
            if j < K - 1:
                out = self.oracle(token, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                current_pending = out.logits[:, -1, :]

        # cache now covers draft_tokens[0..K-2] — the last drafted token
        # was never fed in, so it's not in the cache yet. That's expected.
        return draft_tokens, cache

    def _verify(self, draft_tokens, sovereign_pending, sovereign_cache):
        """
        To check draft_tokens[i], we need the prediction made BEFORE it was
        appended — sovereign_pending for i=0, otherwise the logits row from
        processing draft_tokens[i-1]. Batching draft_tokens[:-1] in one call
        gives us all of those at once.
        """
        K = self.num_speculative_tokens
        check_logits = [sovereign_pending]

        if K > 1:
            batch_input = torch.cat(draft_tokens[:-1], dim=1)
            out = self.sovereign(batch_input, past_key_values=sovereign_cache, use_cache=True)
            cache = out.past_key_values
            for j in range(K - 1):
                check_logits.append(out.logits[:, j, :])
        else:
            cache = sovereign_cache

        accepted_tokens = []
        mismatch_index = None
        for i in range(K):
            target_token = torch.argmax(check_logits[i], dim=-1, keepdim=True)
            if draft_tokens[i].item() == target_token.item():
                accepted_tokens.append(draft_tokens[i])
            else:
                accepted_tokens.append(target_token)
                mismatch_index = i
                break

        return accepted_tokens, mismatch_index, cache

    def _repair_and_trim(self, model, cache, keep_length, last_token):
        """
        Drops any cache entries built from rejected draft tokens, then feeds
        the true last accepted token through once — this both gives it a
        correct cache entry and produces the pending prediction for next round.
        """
        cache = trim_cache(cache, keep_length)
        out = model(last_token, past_key_values=cache, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    def _run_round(self, oracle_cache, sovereign_cache, oracle_pending, sovereign_pending, accepted_len):
        K = self.num_speculative_tokens
        draft_tokens, oracle_cache = self._draft(oracle_pending, oracle_cache)
        accepted_tokens, mismatch_index, sovereign_cache = self._verify(
            draft_tokens, sovereign_pending, sovereign_cache
        )
        num_accepted = len(accepted_tokens)

        # How many of the batched cache entries are still trustworthy —
        # everything up to (but not including) the always-unconfirmed last
        # accepted token, and up to (but not including) any rejected token.
        valid_batch_len = mismatch_index if mismatch_index is not None else (K - 1)
        keep_len = accepted_len + min(num_accepted - 1, valid_batch_len)

        last_token = accepted_tokens[-1]
        oracle_cache, oracle_pending = self._repair_and_trim(self.oracle, oracle_cache, keep_len, last_token)
        sovereign_cache, sovereign_pending = self._repair_and_trim(self.sovereign, sovereign_cache, keep_len, last_token)

        return accepted_tokens, oracle_cache, sovereign_cache, oracle_pending, sovereign_pending

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> int:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        oracle_out = self.oracle(input_ids, use_cache=True)
        sovereign_out = self.sovereign(input_ids, use_cache=True)
        oracle_cache = oracle_out.past_key_values
        sovereign_cache = sovereign_out.past_key_values
        oracle_pending = oracle_out.logits[:, -1, :]
        sovereign_pending = sovereign_out.logits[:, -1, :]

        accepted_len = prompt_len
        n_generated = 0

        while n_generated < max_new_tokens:
            accepted_tokens, oracle_cache, sovereign_cache, oracle_pending, sovereign_pending = self._run_round(
                oracle_cache, sovereign_cache, oracle_pending, sovereign_pending, accepted_len
            )
            accepted_len += len(accepted_tokens)
            n_generated += min(len(accepted_tokens), max_new_tokens - n_generated)

        return n_generated

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        oracle_out = self.oracle(input_ids, use_cache=True)
        sovereign_out = self.sovereign(input_ids, use_cache=True)
        oracle_cache = oracle_out.past_key_values
        sovereign_cache = sovereign_out.past_key_values
        oracle_pending = oracle_out.logits[:, -1, :]
        sovereign_pending = sovereign_out.logits[:, -1, :]

        accepted_len = prompt_len
        generated_ids = []

        while len(generated_ids) < max_new_tokens:
            accepted_tokens, oracle_cache, sovereign_cache, oracle_pending, sovereign_pending = self._run_round(
                oracle_cache, sovereign_cache, oracle_pending, sovereign_pending, accepted_len
            )
            accepted_len += len(accepted_tokens)
            for tok in accepted_tokens:
                if len(generated_ids) >= max_new_tokens:
                    break
                generated_ids.append(tok.item())

        return self.tokenizer.decode(generated_ids)
    

    @torch.no_grad()
    def generate_with_stats(self, prompt: str, max_new_tokens: int):
        """
        Same round loop, but tracks rounds/acceptance instead of being
        optimized purely for timing. Returns (n_generated, avg_accepted_per_round).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        oracle_out = self.oracle(input_ids, use_cache=True)
        sovereign_out = self.sovereign(input_ids, use_cache=True)
        oracle_cache = oracle_out.past_key_values
        sovereign_cache = sovereign_out.past_key_values
        oracle_pending = oracle_out.logits[:, -1, :]
        sovereign_pending = sovereign_out.logits[:, -1, :]

        accepted_len = prompt_len
        n_generated = 0
        num_rounds = 0

        while n_generated < max_new_tokens:
            num_rounds += 1
            accepted_tokens, oracle_cache, sovereign_cache, oracle_pending, sovereign_pending = self._run_round(
                oracle_cache, sovereign_cache, oracle_pending, sovereign_pending, accepted_len
            )
            accepted_len += len(accepted_tokens)
            n_generated += min(len(accepted_tokens), max_new_tokens - n_generated)

        avg_accept = n_generated / num_rounds if num_rounds else 0.0
        return n_generated, avg_accept