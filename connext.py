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
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConnextAgent(Agent):
    ''' The connext agent class 

    Arguments:
        pre_load: the path to the model parameters. If specified, the params is loaded.

    Public attribute:
        connext_net: the neural network of the policy network and value network
        simulations: the number of simulations during the MCTS
        history: a list that record an episode of the gameplay
        batch_size: the size of batch that is going to be fetched from the replay buffer
        lr: learning rate
        prior_baseline: the min value of the prior, this ensure every node in the MCTS have the U value
        optim: optimizer
        mse_loss: loss function for the difference between the actual game result and the value network prediction
        cross_entropy_loss: loss function for the difference between the MCTS result and the policy network prediction
        board_height: the height of the connect4 board
        board_width: the width of the connect4 board
        noisy_steps: the agent will sample actions proportion to their visit_count in the first `noisy_steps` of moves in the game
        
    '''

    def __init__(self, time_per_move=None, pre_load=None, training=True):
        super().__init__()
        self.connext_net = ConnextNet()
        self.simulations = 200
        self.history = []
        self.batch_size = agent_config.config['connext']['batch_size']
        self.lr = agent_config.config['connext']['lr']
        self.prior_baseline = agent_config.config['connext']['prior_baseline']
        self.puct_coef = agent_config.config['connext']['puct_coef']

        self.optim = optim.RMSprop(self.connext_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=0)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.board_height = env_config.config['height']
        self.board_width = env_config.config['width']

        self.noisy_steps = 8

        self.training = training

        self.time_per_move = float('inf')

        if time_per_move is not None:
            self.time_per_move = time_per_move

        if pre_load is not None:
            self.connext_net.load_state_dict(torch.load(pre_load))

    def step(self, board):
        ''' Decide the next move based on the result of MCTS given the board 

        Arguments:
            board: a Board class instance

        Returns: 
            it return the next move made by the agent
        '''
        with torch.no_grad():

            # the root of the search tree
            root = Node(is_root=True, token=self.token)
            
            # run MCTS
            start_time = time.time()
            for i in range(self.simulations):

                root_board = deepcopy(board)
                self.policy_mcts(root, root_board)

                end_time = time.time()
                if end_time - start_time >= self.time_per_move:
                    break

            # calculate the distribution of the action space
            action_count = self.__get_action_count(root)
            action_distribution = action_count / np.sum(action_count)

            # make the move
            deterministic = True if board.steps >= self.noisy_steps or not self.training else False
            action = self.__sample_action(action_distribution, deterministic)

            # construct the board features and store it into the game history(self.history)

            if self.training:
                board_features = self.__construct_features(root, board)
                self.__push_history(board_features, action_distribution)


            return action


    def __push_history(self, board_features, action_distribution):
        ''' Push the board features and the result of MCTS into the game history, it is going to be pushed into the replay buffer in the future.

        Arguments:
            board_features: the torch tensor represent the features
            action_distribution: the distribution of the action space based on the MCTS 
        '''
        self.history.append([board_features, torch.from_numpy(action_distribution)])

    def learn(self, buffer):
        ''' The optimization step that modifies the model params

        It sample the data from the replay buffer, and calculate the loss based on the loss function defined on the paper.

        Arguments:
            buffer: the replay buffer where the training data is from
        '''

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
        ''' Do the MCTS based on the policy network

        Arguments:
            root: a Node class instance that represents the root of the search tree
            board: a Board class instance that represents the current board
        '''

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
        ''' Select the next visited child during MCTS based on the PUCT value

        Arguments: 
            node: A node class instance that represent the parent node

        Returns:
            the move that is made to reach the selected child and the selected child
        '''
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
        ''' An implementaion of PUCT formula, the formula is listed on the Method section in the AlphaGo Zero paper. The PUCT value is based on two major components, the expected reward and the uncertainty.

        Arguments:
            child: A Node class instance that represent the child 
            parent: A Node class instance that represent the parent
        '''
        U = self.puct_coef * (child.prior + self.prior_baseline) * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        Q = child.mean_action_value

        return -Q + U

    def __expand(self, node, board):
        ''' Expand the leaf node(add the children to the leaf node, a child is added only if there is a legel move to transit to that child).

        During the expansion, it evaluate and initialize the child node based on the output of the self.connext_net

        Arguments
            node: A Node class intance that represents the leaf node
            board: A Board class intance that represents the current board  

        Returns:
            returns the expected outcome evaluated by the self.connext_net, the returned value is going to take part in the backpropogation step
        '''
        if board.terminated:
            value = self.get_terminal_value(board.winner_token)
            return value

        # if it is a terminated node, then we should return the actual outcome 

        board_tensor = self.__construct_features(node, board).unsqueeze(0)

        # potential bug: priors on illegal moves make the prior of legal moves not sums up to 1
        priors, value = self.connext_net(board_tensor)
        priors = self.softmax(priors.squeeze())
        value = value.squeeze().item()

        legal_moves = board.get_legal_moves()

        child_token = 1 if node.token == 2 else 2

        if node.is_root and self.training:
            alpha = [0.3 for _ in range(len(legal_moves))]
            noise = np.random.dirichlet(alpha=alpha)

            noise = torch.from_numpy(noise).to(device)
            i = 0
            eta = 0.25
            for legal_move in legal_moves:
                priors[legal_move] = (1 - eta) * priors[legal_move] + eta * noise[i]
                i += 1 


        for legal_move in legal_moves:
            node.children.append(Node(token=child_token, prior=priors[legal_move], last_move=legal_move, parent=node))


        return value

    def get_terminal_value(self, winner_token):
        if winner_token == 0:
            return 0
        else:
            return -1

    def __construct_features(self, node, board):
        ''' Contruct the board features

        The contructed board features is of the form (2, self.board_height, self.board_width).

        第一個channel: 當board上在某(x, y)座標上有與當前玩家有相同的token(node.token)時, 則在feature plane上的(x, y)被mark為1, 其餘座標上的值皆為0
        第二個channel: 當board上在某(x, y)座標上有與當前玩家有不同的token(node.token)時, 則在feature plane上的(x, y)被mark為1, 其餘座標上的值皆為0

        Arguments:
            node: A Node class intance that represents the current node
            board: A Board class intance that represents the current board 

        Returns:
            the constructed board features
        '''

        board_feature = np.zeros((2, self.board_height, self.board_width))

        for i in range(self.board_height):
            for j in range(self.board_width):
                if board.board[i][j] == node.token:
                    board_feature[0][i][j] = 1
                elif board.board[i][j] != 0:
                    board_feature[1][i][j] = 1

        return torch.from_numpy(board_feature).float().to(device)

    def __backpropagate(self, value, node):
        ''' Backup the value of the leaf node all the way up to the root

        Arguments
            value: the updated value from the expansion stage
            node: A Node class intance that represents the current node
        '''
        while node is not None:
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent
            value *= -1


    def __get_action_count(self, root):
        ''' put the visited count of the children of the root into a single numpy array

        Arguments:
            root: the root of the search tree
        Returns:
            returns the visited count of the childrent
        '''

        action_count = np.zeros((self.board_width))
        for child in root.children:
            action_count[child.last_move] = child.visit_count

        return action_count
        

    def update_history(self, result):
        ''' It is called after the game has ended. It update the result of the game to every single entry in the self.history.

        Arguments:
            result: the result of the game from the last player's point of view
        '''
        for i in range(len(self.history) - 1, -1, -1):
            self.history[i].append(result)

            # flip the result, since the players take turns to play
            result *= -1
    
    def clean_history(self):
        ''' clean the outdated history
        '''
        self.history = []

    def set_simulations(self, simulations):
        self.simulations = simulations

    def __sample_action(self, action_distribution, deterministic=False):
        ''' It samples the action based on the action_distribution. There are two versions of the sampling methods, the first one samples the action proportion to the action_distribution, the second one directly sample the action with the largest distribution value.

        Arguments:
            action_distribution: A numpy array that represents the distribution of action space
            deterministic: the sampling type

        Returns: 
            the sampled action
        '''

        if deterministic:
            return np.argmax(action_distribution)
        
        cumulative_distribution = np.cumsum(action_distribution, axis=0)
        point = np.random.uniform(0, 1)

        for idx, d in enumerate(cumulative_distribution):
            if d >= point:
                return idx 
        

    def make_random_move(self, board):
        legal_moves = board.get_legal_moves()

        # clean the history, since the outcome cannot reflect the effect of previous moves anymore
        self.clean_history()
        return np.random.choice(legal_moves)


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