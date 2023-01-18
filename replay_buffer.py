from collections import deque
from replay_buffer_config import config
import random
import torch
import math

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=5)

    def append_dataset(self, game_history):
        self.buffer.append(game_history)

    def sample(self, batch_size=config['batch_size']):
        batch = []

        for i in range(len(self.buffer)):
            sample_size = batch_size // len(self.buffer)
            if i == len(self.buffer) - 1:
                sample_size += batch_size % len(self.buffer)
            
            batch += random.sample(self.buffer[i], sample_size)

        board_features = []
        action_distributions = []
        results = []

        for experience in batch:
            # add back the augment if the training is stable
            augmented_experience = self.augment(experience)
            board_features.append(augmented_experience[0]) 
            action_distributions.append(augmented_experience[1])
            results.append([augmented_experience[2]])

        board_features = torch.stack(board_features)
        action_distributions = torch.stack(action_distributions)
        results = torch.Tensor(results)

        return board_features, action_distributions, results

    def augment(self, experience):
        horizontal_flip_random = random.random() > 0.5
        board_features, action_distribution, result = experience

        flipped_board = board_features.clone()
        flipped_action_distribution = action_distribution.clone()

        # print(f'original board\n: {board_features}, action_distributions: {action_distributions}')

        if horizontal_flip_random:
            flipped_board = torch.flip(flipped_board, dims=[2])
            flipped_action_distribution = torch.flip(flipped_action_distribution, dims=[0])


        # print(f'augmented board after hori_flip: {horizontal_flip_random}, result_flip: {result_flip_random}:\n {board_features}, action_distribution: {action_distributions}')

        return flipped_board, flipped_action_distribution, result


    def clean_buffer(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]