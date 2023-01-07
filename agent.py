import numpy as np

class Agent:
    def __init__(self):
        self.token = None

    def step(self, board):
        pass

    def get_legal_moves(self, board):
        legal_moves = [1 if token == 0 else 0 for token in board[0]] 
        return legal_moves
    

class Human(Agent):
    def __init__(self):
        super().__init__()

    def step(self, board):
        return int(input('Next move: '))
    
class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def step(self, board):
        legal_moves = self.get_legal_moves(board)
        legal_moves = [i for i in range(len(legal_moves)) if legal_moves[i] == 1]
        return np.random.choice(legal_moves)

