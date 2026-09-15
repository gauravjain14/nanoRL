import torch
import torch.nn.functional as F


def token_logprobs(logits, input_ids):
    log_probs = F.log_softmax(
        logits[:, :-1], dim=-1, dtype=torch.float32)
    target_ids = input_ids[:, 1:]
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
