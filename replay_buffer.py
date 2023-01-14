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
            board_features.append(experience[0]) 
            action_distributions.append(experience[1])
            results.append([experience[2]])

        board_features = torch.stack(board_features)
        action_distributions = torch.stack(action_distributions)
        results = torch.Tensor(results)

        return board_features, action_distributions, results

    def horizontal_flip(self, board):

        board = torch.flip(board, dims=[2]) if random.uniform(0, 1) >= 0.5 else board # dim 1 is channel
        return board

    def augment(self, board):
        board = self.horizontal_flip(board)

    def clean_buffer(self):
        self.buffer.clear()