import numpy as np
from agent_config import config
from MCTS import naive_mcts

class Agent:
    def __init__(self):
        self.token = None

    def step(self, board):
        pass

    def get_legal_moves(self, board):
        '''
        Return a list of indices of legal moves
        '''
        legal_moves = [1 if token == 0 else 0 for token in board[0]] 
        legal_moves = [i for i in range(len(legal_moves)) if legal_moves[i] == 1]
        return legal_moves

    def flip_board(self, board):
        '''
        This function flip the tokens on the board(1 -> 2, 2 -> 1).

        To simplify the training process, the some agents always assume it plays the token 1. If self.token is 2, then the flip_board() function call is neccessary.

        The flip_board function will be called by the child class in the beginning of their step function call.
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
    def __init__(self):
        super().__init__()

    def step(self, board):
        self.flip_board(board)
        return int(input('Next move: '))
    

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def step(self, board):
        legal_moves = self.get_legal_moves(board)
        return np.random.choice(legal_moves)


class MCTSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.config = config['mcts_agent']
        # self.rollout_policy = self.config['rollout_policy']

    def step(self, board):
        # assume the rollout are composed of random gameplay now

        return naive_mcts(board, self.token)
        