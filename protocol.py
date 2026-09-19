

class RolloutGroup:
    def __init__(self, group_id, attempt_id, run_epoch,
            source_learner_step, created_at, trajectories):
        """
            source_learner_step: The learner step that the rollout group
            was created at.
        """
        self.group_id = group_id
        self.attempt_id = attempt_id
        self.run_epoch = run_epoch
        self.created_at = created_at
        self.trajectories = trajectories
        self.source_learner_step = source_learner_step

    
class Trajectory:
    def __init__(self, prompt_ids, response_ids, behavior_logprobs,
                 loss_mask, reward, finish_reason):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        # not sure why they are called this but this is comparing
        # trainer logprobs to the generator policy's logprobs.
        self.behavior_logprobs = behavior_logprobs
        # loss_mask - which tokens contribute to the loss.
        self.loss_mask = loss_mask
        self.reward = reward
        self.finish_reason = finish_reason