from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardLogger, self).__init__(logdir)

    def log_training(self, steps, loss, lr, value_policy, value_target=None, epsilon=None):
        self.add_scalar('Loss/train', loss, steps)
        self.add_scalar('Expected_value/policy_network', value_policy, steps)
        if value_target is not None:
            self.add_scalar('Expected_value/target_network', value_target, steps)
        self.add_scalar('lr', lr, steps)
        if epsilon is not None:
            self.add_scalar('epsilon', epsilon, steps)

    def log_episode(self, episode, avg_loss, score):
        self.add_scalar('Episode/score', score, episode)
        self.add_scalar('Episode/avg_loss', avg_loss, episode)
