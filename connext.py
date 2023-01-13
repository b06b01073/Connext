import torch
from torch import nn
from env_config import config
from copy import deepcopy
import math
import random
from collections import deque
import torch
import numpy as np

class ConnextAgent():
    def __init__(self):
        super().__init__()
        self.connext_net = ConnextNet()
        self.simulations = 3
        self.fake_token = 1
        self.game_history = []

    def step(self, board):

        if self.token == 2:
            self.__flip_board(board.board)

        root = Node()
        
        for _ in range(self.simulations):
            root_board = deepcopy(board)
            self.policy_mcts(root, root_board)

        action_distribution = self.__get_action_distribution(root)
        self.game_history.append([board, action_distribution])

        return np.argmax(action_distribution)

    def __flip_board(self, board):
        
        width = config['width']
        height = config['height']

        for i in range(height):
            for j in range(width):
                if board[i][j] == 1:
                    board[i][j] = 2
                elif board[i][j] == 2:
                    board[i][j] = 1

        return board

    def learn(self):
        # sample mini batch from replay buffer and update 
        pass

    def policy_mcts(self, root, board):
        node = root
        node.visit_count += 1

        # run Select until a leaf node is met
        while not node.is_leaf():
            action, node = self.__select_child(node)


            # flip the board and always play 1
            self.__flip_board(board.board)

            # we can always assume the step function call on a non terminal state
            board.step(action, self.fake_token)

        # Expand and evaluate
        value = self.__expand(node, board)

        # Backup
        self.__backpropagate(value, node)


            
    def __select_child(self, node):
        max_PUCT = float('-inf')
        selected_child = None

        for child in node.children:
            PUCT_score = self.__get_PUCT(child, node)
            if PUCT_score > max_PUCT:
                selected_child = child
                max_PUCT = PUCT_score

        selected_child.visit_count += 1
        return selected_child.last_move, selected_child

    def __get_PUCT(self, child, parent):
        U = child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        Q =  child.mean_action_value
        return Q + U

    # def __get_ucb(self):
        


    def __expand(self, node, board):
        board_tensor = torch.from_numpy(board.board).unsqueeze(0).unsqueeze(0).float()
        priors, value = self.connext_net(board_tensor)
        priors.squeeze_()
        value.squeeze_()

        legal_moves = board.get_legal_moves()

        if board.terminated:
            return value

        

        # the terminal node cannot expand
        for legal_move in legal_moves:
            node.children.append(Node(prior=priors[legal_move], last_move=legal_move, parent=node))

        return value


    def __backpropagate(self, value, node):
        while node is not None:
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent


    def __get_action_distribution(self, root):
        action_distribution = np.zeros((config['width']))
        for child in root.children:
            action_distribution[child.last_move] = child.visit_count

        action_distribution /= np.sum(action_distribution)

        return action_distribution
        

    # def __play(self, root):

    #     move = None
    #     max_visit_count = -1
    #     width = config['width']

    #     action_distribution = [0 for i in range(width)]

    #     for child in root.children:
    #         if child.visit_count > max_visit_count:
    #             move = child.last_move 
    #             max_visit_count = child.visit_count

    #     return move    


class Node:
    def __init__(self, prior=None, last_move=None, parent=None):

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior = prior
        self.children = []
        self.last_move = last_move
        self.parent = parent

    def is_leaf(self):
        return len(self.children) == 0


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=1000)

    def append(self, game_record):

        for record in game_record:
            print(record)
            self.buffer.append(record)

        

class ConnextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 1

        self.width = config['width']

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.policy_network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.width),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )


        self.value_network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )



    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        
        action_distribution = self.policy_network(x)
        value = self.value_network(x)

        return action_distribution, value


    
        