import torch
from torch import nn
import env_config
import agent_config
from copy import deepcopy
import math
import random
import torch
import numpy as np
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConnextAgent():
    def __init__(self):
        super().__init__()
        self.connext_net = ConnextNet()
        self.simulations = 500
        self.fake_token = 1
        self.history = []
        self.batch_size = agent_config.config['connext']['batch_size']
        self.lr = agent_config.config['connext']['lr']
        self.optim = optim.RMSprop(self.connext_net.parameters(), lr=self.lr, weight_decay=0.01)
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=0)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def step(self, board):
        with torch.no_grad():
            if self.token == 2:
                self.__flip_board(board.board)

            root = Node()
            
            for _ in range(self.simulations):
                root_board = deepcopy(board)
                self.policy_mcts(root, root_board)

            action_distribution = self.__get_action_distribution(root)

            action = np.argmax(action_distribution)

            self.__push_history(board.board.astype('float'), action_distribution)

            return action

    def __push_history(self, board, action_distribution):
        self.history.append([torch.from_numpy(board).float().unsqueeze(0), torch.from_numpy(action_distribution)])

    def __flip_board(self, board):
        
        width = env_config.config['width']
        height = env_config.config['height']

        for i in range(height):
            for j in range(width):
                if board[i][j] == 1:
                    board[i][j] = 2
                elif board[i][j] == 2:
                    board[i][j] = 1

        return board

    def learn(self, buffer):
        if len(buffer.buffer) < self.batch_size:
            return

        game_positions, action_distribution_labels, result_labels = buffer.sample(self.batch_size)
        game_positions = game_positions.to(device)
        action_distribution_labels = action_distribution_labels.to(device)
        result_labels = result_labels.to(device)


        policy_network_preds, result_preds = self.connext_net(game_positions)

        policy_network_preds = policy_network_preds.to(device)
        result_preds = result_preds.squeeze().to(device)


        # print(f'Shapes:\n result_labels: {result_labels.shape}, result_preds: {result_preds.shape}\n policy_network_preds: {policy_network_preds.shape}, action_distribution_labels: {action_distribution_labels.shape}')


        self.optim.zero_grad()
        mse_loss = self.mse_loss(result_labels, result_preds)
        cross_entropy_loss = self.cross_entropy_loss(policy_network_preds, action_distribution_labels) 
        loss = mse_loss + cross_entropy_loss
        loss.backward()
        self.optim.step() 

        return loss



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


    def __expand(self, node, board):
        board_tensor = torch.from_numpy(board.board).unsqueeze(0).unsqueeze(0).float().to(device)
        priors, value = self.connext_net(board_tensor)
        priors = self.softmax(priors.squeeze())
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
        action_distribution = np.zeros((env_config.config['width']))
        for child in root.children:
            action_distribution[child.last_move] = child.visit_count

        action_distribution /= np.sum(action_distribution)

        return action_distribution
        

    def update_history(self, winner_token):
        for i in range(len(self.history)):
            token = i % 2 + 1
            if winner_token == 0:
                self.history[i].append(0)
            elif winner_token == token:
                self.history[i].append(1)
            else:
                self.history[i].append(-1)

    
    def clean_history(self):
        self.history = []
            

    # def loss_fn(self, action_distribution_labels, result_labels, policy_network_preds, result_preds):



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

        

class ConnextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 1

        self.width = env_config.config['width']

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU()
        ).to(device)

        self.policy_network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.width),
            nn.ReLU(),
        ).to(device)


        self.value_network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)



    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)

        action_distribution = self.policy_network(x)
        value = self.value_network(x)

        return action_distribution, value


    
        