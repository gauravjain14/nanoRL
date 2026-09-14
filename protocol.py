

class RolloutGroup:
    def __init__(self, group_id, attempt_id, run_epoch,
            source_learner_step, created_at, trajectories):
        """
            source_learner_step: The learner step that the rollout group was created at.
        """
        self.group_id = group_id
        self.attempt_id = attempt_id
        self.run_epoch = run_epoch
        self.created_at = created_at
        self.trajectories = trajectories
        self.source_learner_step = source_learner_step
