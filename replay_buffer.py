from collections import deque
from replay_buffer_config import config
import random
import torch

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()

    def append_history(self, history):
        for experience in history:
            self.buffer.append(experience)

    def sample(self, batch_size=config['batch_size']):
        assert len(self.buffer) >= batch_size, 'Not enough data!'
        batch = random.sample(self.buffer, batch_size)

        board_features = []
        action_distributions = []
        results = []

        for experience in batch:
            # add back the augment if the training is stable
            # experience = self.augment(experience)
            board_features.append(experience[0]) 
            action_distributions.append(experience[1])
            results.append([experience[2]])

        board_features = torch.stack(board_features)
        action_distributions = torch.stack(action_distributions)
        results = torch.Tensor(results)

        return board_features, action_distributions, results

    def augment(self, experience):
        horizontal_flip_random = random.random() > 0.5
        board_features, action_distributions, result = experience

        # print(f'original board\n: {board_features}, action_distributions: {action_distributions}')

        if horizontal_flip_random:
            board_features = torch.flip(board_features, dims=[2])
            action_distributions = torch.flip(action_distributions, dims=[0])

        # print(f'augmented board after hori_flip: {horizontal_flip_random}, result_flip: {result_flip_random}:\n {board_features}, action_distributions: {action_distributions}')

        return board_features, action_distributions, result


    def clean_buffer(self):
        self.buffer.clear()