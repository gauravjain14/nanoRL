import torch
import torch.nn.functional as F


def token_logprobs(logits, input_ids):
    log_probs = F.log_softmax(
        logits[:, :-1], dim=-1, dtype=torch.float32)
    target_ids = input_ids[:, 1:]
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def group_advantages(rewards, group_index, normalize_std=True, eps=1e-6):
    advantages = torch.zeros_like(rewards)

    # assumes group_index is a tensor of shape [rollout_groups]
    for group_id in group_index.unique():
        selected = group_index == group_id
        values = rewards[selected]
        centered = values - values.mean()
        if normalize_std:
            centered = centered / (centered.std() + eps)
        advantages[selected] = centered

    return advantages