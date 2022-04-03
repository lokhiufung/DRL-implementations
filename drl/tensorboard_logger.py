from torch.utils.tensorboard import SummaryWriter

# from plotting_utils import plot_weights_to_numpy


class TensorboardLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardLogger, self).__init__(logdir)

    # def log_training(self, steps, loss, lr, value_policy, value_target=None, epsilon=None):
    #     self.add_scalar('Loss/train', loss, steps)
    #     self.add_scalar('Expected_value/policy_network', value_policy, steps)
    #     if value_target:
    #         self.add_scalar('Expected_value/target_network', value_target, steps)
    #     self.add_scalar('lr', lr, steps)
    #     if epsilon:
    #         self.add_scalar('epsilon', epsilon, steps)
    
    def log_scalar(self, iteration, train_data):
        """
        train_data: dict, key is name of data 
        """
        for key, value in train_data.items():
            self.add_scalar(key, value, iteration)

    # def log_image(self, iteration, train_data):
    #     """
    #     weights: numpy array
    #     """
    #     for key, value in train_data.items():
    #         self.add_image(key, , global_step=steps, dataformats='HWC')

    # def log_episode(self, episode, score, avg_loss=None):
    #     self.add_scalar('Episode/score', score, episode)
    #     if avg_loss:
    #         self.add_scalar('Episode/avg_loss', avg_loss, episode)
