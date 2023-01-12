import torch
from torch import nn
from env_config import config
from copy import deepcopy
import math
import random

class ConnextAgent():
    def __init__(self):
        super().__init__()
        self.connext_net = ConnextNet()
        self.simulations = 10

    def step(self, board):

        legal_move = board.get_legal_moves()
        return random.choice(legal_move)

        # root = Node()
        
        # for _ in range(self.simulations):
        #     root_board = deepcopy(board)
        #     policy_mcts(root, root_board)


        # action_distribution, z = policy_mcts()
        # self.buffer.append([board, action_distribution, z])

        # pick the action

        # return 

    def learn(self):
        # sample mini batch from replay buffer and update 
        pass

    def policy_mcts(self, root, board):
        node = root
        _ = self.__expand(root, board)

        # search until meet a leaf node
        last_move = None
        while not node.is_leaf:
            action, node = __select_child(node)

            # we can always assume the step function call is valid
            board.step(action)
        value = self.__expand()

            
    def __select_child(self, node):
        max_PUCT = float('-inf')
        selected_child = None

        for child in node.children:
            PUCT_score = __get_PUCT(child, node)
            if PUCT_score > max_PUCT:
                selected_child = child
                max_PUCT = PUCT_score

        return selected_child.last_move, selected_child

    def __get_PUCT(self, child, parent):
        U = child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        Q =  child.mean_action_value
        return Q + U

    # def __get_ucb(self):
        


    # def __expand(self, node, board):
    #     priors, value = self.connext_net(board)

    #     legal_moves = board.get_legal_moves()

    #     # the terminal node cannot expand
    #     for legal_move in legal_moves:
    #         node.children.append(Node(prior=priors[legal_move], last_move=legal_move))

    #     return value






class ReplayBuffer:
    def __init__(self):
        pass
        

class ConnextNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(

        )

class Node:
    def __init__(self, prior=None, last_move=None):
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior = prior
        self.children = []
        self.last_move = last_move

    def is_leaf(self):
        return len(self.children) == 0

    
        