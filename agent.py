import numpy as np
from agent_config import config
from MCTS.MCTS import naive_mcts
from copy import deepcopy

class Agent:
    ''' The Agent class is the base class of agents in the RL framework.

    Public Attributes:
        token: An integer in {1, 2} that represents the token of the agent
    '''

    def __init__(self):
        self.token = None

    def step(self, board):
        ''' The base function of step
        '''
        pass

    def get_legal_moves(self, board):
        ''' Get legal moves of the given Board class instance

        This Function should be called in the very beginning of step() of the agent
        
        Arguments:
            board: a Board class intances

        Returns:
            A list of legal moves, each entry in the list represent a playable column in the board. 
        '''
        legal_moves = [1 if token == 0 else 0 for token in board[0]] 
        legal_moves = [i for i in range(len(legal_moves)) if legal_moves[i] == 1]
        return legal_moves

    def flip_board(self, board):
        '''  This function flip the tokens on the board(1 -> 2, 2 -> 1).  

        To simplify the training process, the some agents always assume it plays the token 1. If self.token is 2, then the flip_board() function call is neccessary.

        The flip_board function will be called by the child class in the beginning of their step function call.

        Arguments:
            board: A Board class instance

        Returns:
            Return a Board class instance which is the flipped result of the board parameter.
        '''

        flipped_board = np.zeros((board.height, board.width), dtype='uint8')
        one_indices = np.where(board.board == 1)
        two_indices = np.where(board.board == 2)

        # flip 1 to 2
        for row, col in zip(one_indices[0], one_indices[1]):
            flipped_board[row][col] = 2

        # flip 2 to 1
        for row, col in zip(two_indices[0], two_indices[1]):
            flipped_board[row][col] = 1

        return flipped_board
    

class Human(Agent):
    ''' An agent represents a human player by asking input from terminal in every step
    '''

    def __init__(self):
        super().__init__()

    def step(self, board):
        self.flip_board(board)
        return int(input('Next move: '))
    

class RandomAgent(Agent):
    ''' An agent that follows the random policy in decision making 
    '''
    def __init__(self):
        super().__init__()

    def step(self, board):
        ''' Make the next move based on the policy of the agent

        Arguments:
            board: A board class instance

        Returns:
            A move picked randomly from legal moves.
        '''
        legal_moves = self.get_legal_moves(board)
        return np.random.choice(legal_moves)


class MCTSAgent(Agent):
    ''' An agent that make decisions based on the result of Monte Carlo tree search(MCTS)

    Public Attributes:
        simulations: the number of times of the simulation in the MCTS phase
    '''

    def __init__(self, simulations):
        super().__init__()
        self.simulations = simulations

    def step(self, board):
        ''' Make the next move based on the policy of the agent

        Arguments:
            board: A board class instance

        Returns:
            An integer given by the result of MCTS that represents the next move.
        '''
        return naive_mcts(board, self.token, self.simulations, self.rollout_policy)


    def rollout_policy(self, node):
        ''' The policy that the rollout stage is going to follow

        The rollout policy is a random policy in this version.

        Arguments:
            node: A Node class instance that represents the starting point of rollout.

        Returns:
            An integer that represents the token of the winner after the rollout.         
        '''

        board = deepcopy(node.state)
        token = node.token
        while True:
            legal_moves = board.get_legal_moves()

            move = np.random.choice(legal_moves)
            board.step(move, token)

            if board.terminated:
                break

            if token == 1:
                token = 2
            else:
                token = 1
            
        return board.winner_token
            