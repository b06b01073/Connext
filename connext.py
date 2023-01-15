import torch
from torch import nn
import env_config
import agent_config
from copy import deepcopy
import math
import torch
import numpy as np
from torch import optim
from model import ConnextNet
from agent import Agent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConnextAgent(Agent):
    def __init__(self, pre_load=None):
        super().__init__()
        self.connext_net = ConnextNet()
        self.simulations = 200
        self.history = []
        self.batch_size = agent_config.config['connext']['batch_size']
        self.lr = agent_config.config['connext']['lr']
        self.prior_baseline = agent_config.config['connext']['prior_baseline']

        self.optim = optim.RMSprop(self.connext_net.parameters(), lr=self.lr, weight_decay=1e-3)
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=0)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.board_height = env_config.config['height']
        self.board_width = env_config.config['width']

        if pre_load is not None:
            self.connext_net.load_state_dict(torch.load(pre_load))

    def step(self, board):
        with torch.no_grad():

            root = Node(is_root=True, token=self.token)
            
            for _ in range(self.simulations):
                root_board = deepcopy(board)
                self.policy_mcts(root, root_board)

            action_count = self.__get_action_count(root)
            action_distribution = action_count / np.sum(action_count)

            deterministic = True if board.steps >= 5 else False
            action = self.__sample_action(action_distribution, deterministic)

            board_tensor = self.__construct_features(root, board)
            self.__push_history(board_tensor, action_distribution)

            return action


    def __push_history(self, board_tensor, action_distribution):
        self.history.append([board_tensor, torch.from_numpy(action_distribution)])

    def learn(self, buffer):
        game_positions, action_distribution_labels, result_labels = buffer.sample(self.batch_size)


        game_positions = game_positions.to(device)
        action_distribution_labels = action_distribution_labels.to(device)
        result_labels = result_labels.to(device)

        policy_network_preds, result_preds = self.connext_net(game_positions)

        policy_network_preds = policy_network_preds.to(device)
        result_preds = result_preds.to(device)


        # print(f'Shapes:\n result_labels: {result_labels.shape}, result_preds: {result_preds.shape}\n policy_network_preds: {policy_network_preds.shape}, action_distribution_labels: {action_distribution_labels.shape}')


        self.optim.zero_grad()
        mse_loss = self.mse_loss(result_labels, result_preds)
        cross_entropy_loss = self.cross_entropy_loss(policy_network_preds, action_distribution_labels) 
        loss = mse_loss + cross_entropy_loss

        # print(f'mse loss: {mse_loss:.3f}, cross entropy: {cross_entropy_loss: .3f}')
        loss.backward()
        self.optim.step() 

        return mse_loss, cross_entropy_loss


    def policy_mcts(self, root, board):
        node = root
        node.visit_count += 1

        # run Select until a leaf node is met
        while not node.is_leaf():
            token = node.token
            action, node = self.__select_child(node)
            board.step(action, token)

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
        U = (child.prior + self.prior_baseline) * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        Q = child.mean_action_value

        return Q + U

    def __expand(self, node, board):
        board_tensor = self.__construct_features(node, board).unsqueeze(0)
        priors, value = self.connext_net(board_tensor)
        priors = self.softmax(priors.squeeze())
        value.squeeze_()

        legal_moves = board.get_legal_moves()

        child_token = 1 if node.token == 2 else 2

        # the terminal node cannot expand
        for legal_move in legal_moves:
            noise = 0
            if node.is_root:
                noise += np.abs(np.random.normal(scale=0.05))
            node.children.append(Node(token=child_token, prior=priors[legal_move] + noise, last_move=legal_move, parent=node))

        if node.token != self.token:
            value *= -1

        return value

    def __construct_features(self, node, board):
        board_feature = np.zeros((2, self.board_height, self.board_width))

        for i in range(self.board_height):
            for j in range(self.board_width):
                if board.board[i][j] == node.token:
                    board_feature[0][i][j] = 1
                elif board.board[i][j] != 0:
                    board_feature[1][i][j] = 1

        return torch.from_numpy(board_feature).float().to(device)

    def __backpropagate(self, value, node):
        while node is not None:
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent


    def __get_action_count(self, root):
        action_count = np.zeros((self.board_width))
        for child in root.children:
            action_count[child.last_move] = child.visit_count

        return action_count
        

    def update_history(self, winner_token):
        for i in range(len(self.history)):
            player_token = i % 2 + 1
            if winner_token == 0:
                self.history[i].append(0)
            elif winner_token == player_token:
                self.history[i].append(1)
            else:
                self.history[i].append(-1)

    
    def clean_history(self):
        self.history = []
            

    def __sample_action(self, action_distribution, deterministic=False):

        if deterministic:
            return np.argmax(action_distribution)
        
        cumulative_distribution = np.cumsum(action_distribution, axis=0)
        point = np.random.uniform(0, 1)

        for idx, d in enumerate(cumulative_distribution):
            if d >= point:
                return idx 
        



class Node:
    def __init__(self, token, is_root=False, prior=None, last_move=None, parent=None):

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior = prior
        self.children = []
        self.last_move = last_move
        self.parent = parent
        self.is_root = is_root
        self.token = token

    def is_leaf(self):
        return len(self.children) == 0