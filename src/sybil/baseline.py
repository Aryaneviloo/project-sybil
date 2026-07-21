import torch


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int,
                       device: str) -> int:
    """
    Returns the number of tokens generated. 
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    past_key_values = None
    current_input = input_ids
    n_generated = 0

    for _ in range(max_new_tokens):
        outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        # After the first step, we only ever need to feed the newest token —
        # the cache already holds everything before it.
        current_input = next_token
        n_generated += 1

    return n_generated
