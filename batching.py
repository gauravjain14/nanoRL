



def collate_groups(groups, pad_token_id):
    """ flattent the groups into one row per trajectory.
    input_ids = [B, T] (prompt + response and right padded)
    attention_mask = [B, T] (prompt + response and right padded)
    loss_mask = [B, T] (false for prompt and padding)
    behavior_logprobs = [B, T]
    rewards = [B]
    group_index = [B]
    """
    pass