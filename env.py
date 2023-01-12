import numpy as np
from env_config import config
from termcolor import cprint, colored
from copy import deepcopy

class ConnectX:
    ''' The environment of the RL framework.

    Public Attributes:
        board: a Board class instance that represent the connectX board
        embedded_player: a Agent class instance that represents an agent which is the opponent of the real agent
        embedded_player_token: an integer that represents the token of the embedded player
        agent: a Agent class instance that represents the real agent
        agent_token: an integer that represents the token of the agent
        terminated: a boolean that if True if the game has ended
        first: an integer that represent which player goes first 
        last_embedded_player_move: a integer tuple that represent the last move played by the embedded_player
        last_agent_move: a integer tuple that represent the last move played by the agent
    '''
    def __init__(self):
        self.board = Board()
        self.embedded_player = None
        self.embedded_player_token = None
        self.agent = None
        self.agent_token = None
        self.terminated = False

        self.first = None # embedded_player goes first if self.first == 0 else self.agent goes first

        # record the last move for better UI
        self.last_embedded_player_move = None
        self.last_agent_move = None

    def step(self, col):
        ''' The agent calls ConnectX.step(col) will put an token at the col-th column.

        Arguments:
            col: an integer that represents the move made by the caller

        Returns:
            Returns an integer that represents the board and result of the game, the result of the game is defined as follow:
                1, if self.agent won
                -1, if self.embedded_player won
                0, otherwise
        '''
        self.last_agent_move = self.board.step(col, self.agent_token)
        if self.board.has_winner:   # the self.agent won by making the move 
            self.terminated = True
            return self.board, 1, self.terminated


        if self.board.draw:
            self.terminated = True
            return self.board, 0, self.terminated

        col = self.embedded_player.step(self.board)
        self.last_embedded_player_move = self.board.step(col, self.embedded_player_token)
        if self.board.has_winner:   # the self.embedded_player won by making the move
            self.terminated = True
            return self.board, -1, self.terminated
        
        # non-terminal state, we should always return an copy to prevent agent modifies the internel state of the env

        if self.board.draw:
            self.terminated = True
            return self.board, 0, self.terminated

        return deepcopy(self.board), 0, self.terminated
        

    def register(self, agent):
        ''' The ConnectX.register(agent) call register the agent as self.agent.
        
        Once an agent registerd in the env, the env will decide who goes first by the self.pick_first() function call. The player goes first has token of 1, the player goes second has token of 2.

        If the agent goes second, then this function call forced the embedded_agent make a move immediatly.


        Arguments: 
            agent: an Agent class instance that represents the registering agent

        Returns:
            This function call returns the token of self.agent and the first observation of the board.
            
        '''

        self.agent = agent

        self.pick_first()

        # the embedded_agent plays the first move
        if self.first == 0:
            col = self.embedded_player.step(self.board)
            self.last_embedded_player_move = self.board.step(col, self.embedded_player_token)
        
        # self.display_start()

        return self.agent_token, deepcopy(self.board)

    def pick_first(self):
        ''' decide the first player of the game
        '''
        self.first = np.random.randint(low=0, high=2)

        if self.first == 0:
            self.embedded_player_token = 1
            self.embedded_player.token = 1
            self.agent_token = 2
        else:
            self.embedded_player_token = 2
            self.embedded_player.token = 2
            self.agent_token = 1

    def display_start(self):
        ''' Log the starting message 

        Display the starting position from the agent's point of view. If the agent goes first, then the starting board should be empty. If the embedded_player goes first, then the starting board should have one token.
        '''
        print('The game has started!')
        first_player = 'embedded_player' if self.first == 0 else 'agent'
        second_player = 'embedded_player' if self.first != 0 else 'agent'

        first_token = colored('1', color='red') if self.first == 0  else colored('1', color='green')
        second_token = colored('2', color='red') if self.first != 0 else colored('2', color='green')

        print(f'{first_player} goes first with token {first_token}, {second_player} goes second with token {second_token}')

        print('The starting position for agent: ')
        self.render()

    def render(self):
        ''' print the board
        '''
        self.board.render(self.last_embedded_player_move, self.last_agent_move, self.embedded_player_token, self.agent_token)

class Board:
    ''' This class represent the board of the connectX game

    Public Attributes:
        height: an integer that represents the height of the board 
        width: an integer that represents the width of the board 
        X: an integer that represents the number of token a player has to connect in order to win
        board: a 2D nparray that represents the board
        terminated: a boolean that if True if the game has ended
        steps: an integer that represent how many steps has taken in the game
        winner_token: an integer that represent the winner of the game

    '''

    def __init__(self):
        self.height = config['height']
        self.width = config['width']
        self.X = config['X']
        self.board = np.zeros((self.height, self.width), dtype='uint8')
        self.terminated = False
        self.steps = 0
        self.has_winner = False
        self.draw = False

        self.winner_token = 0 # 0 if draw, 1 if player with token 1 won, 2 if player with token 2 won

    def step(self, col, token):
        ''' Play a move on the board
        The ConnectX env put an token on the col-th column, and it also check whether the move ends the game. 

        Arguments:
            col: an integer that represent column of the move
            token: an integer that represent the player

        Returns:
            A tuple that represents the coordinates of the final position of the played token.
        '''
        assert col >= 0 and col < self.width and self.board[0][col] == 0, 'Illegal move!'

        for row in range(self.height):
            if row == self.height - 1 or self.board[row + 1][col] != 0:
                self.board[row][col] = token

                # warning: step need to increment before the __is_end function call
                self.steps += 1
                self.terminated = self.__is_end(row, col, token)
                return row, col



    def __is_end(self, row, col, token):
        ''' The __is_end uses coord (row, col) as a center to check the winning condition

        The returned value `terminated` is decided base on two conditions:
            1. Whether there is a winner
            2. Whether the game is a draw        
        '''

        self.has_winner = self.__check_horizontal(row, col, token) or self.__check_vertical(row, col, token) or self.__check_diagonal(row, col, token)
        if not self.has_winner:
            self.draw = (self.steps == self.width * self.height)

        if self.has_winner:
            self.winner_token = token

        return self.has_winner or self.draw

    def __check_horizontal(self, row, col, token):
        connected_tokens = 1
        
        # check rightward
        for i in range(col + 1, self.width):
            if self.board[row][i] == token:
                connected_tokens += 1
            else:
                break

        # check leftward
        for i in reversed(range(col)):
            if self.board[row][i] == token:
                connected_tokens += 1
            else:
                break

        return True if connected_tokens >= self.X else False

    def __check_vertical(self, row, col, token):
        connected_tokens = 1

        # check downward
        for i in range(row + 1, self.height):
            if self.board[i][col] == token:
                connected_tokens += 1
            else:
                break

        # check upward
        for i in reversed(range(row)):
            if self.board[i][col] == token:
                connected_tokens += 1
            else:
                break

        
        return True if connected_tokens >= self.X else False

    def __check_diagonal(self, row, col, token):
        terminated = self.__upright_downleft(row, col, token) or self.__upleft_downright(row, col, token) 


        return True if terminated else False
    
    def __upright_downleft(self, row, col, token):
        connected_tokens = 1
        i = 1

        # upright
        while col + i < self.width and row - i >= 0:
            if self.board[row - i][col + i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

        # downleft
        i = 1
        while col - i >= 0 and row + i < self.height:
            if self.board[row + i][col - i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

        # if connected_tokens >= self.X:
        #     print('__upright_downleft')

        return True if connected_tokens >= self.X else False
        
    def __upleft_downright(self, row, col, token):  
        connected_tokens = 1
        i = 1

        # upleft
        while col - i >= 0 and row - i >= 0:
            if self.board[row - i][col - i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

        i = 1
        # downright
        while col + i < self.width and row + i < self.height:
            if self.board[row + i][col + i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

        # if connected_tokens >= self.X:
        #     print('__upleft_downright')
        return True if connected_tokens >= self.X else False
    
    
    def get_legal_moves(self):
        '''
        Return a list of indices of legal moves
        '''
        legal_moves = [1 if token == 0 else 0 for token in self.board[0]] 
        legal_moves = [i for i in range(len(legal_moves)) if legal_moves[i] == 1]
        return legal_moves

    def render(self, last_embedded_player_move, last_agent_move, embedded_player_token, agent_token):
        '''
        Print the board position with color and bold text
        '''

        for i in range(self.width + 2):
            print('*', end=' ')
        print('')
        for i in range(self.height):
            print('*', end= ' ')
            for j in range(self.width):
                if self.board[i][j] == embedded_player_token:
                    if (i, j) == last_embedded_player_move:
                        print('\033[1m', end='')
                    cprint(self.board[i][j], color='red', end=' ')
                elif self.board[i][j] == agent_token:
                    if (i, j) == last_agent_move:
                        print('\033[1m', end='')
                    cprint(self.board[i][j], color='green', end=' ')
                else:
                    print(self.board[i][j], end=' ')
            print('*')


        print('*', end=' ')
        for _ in range(self.width):
            print('-', end=' ')

        print('*')
        print('*', end=' ')
        for i in range(self.width):
            cprint(i, color='cyan', end=' ')
        print('*')

        for i in range(self.width + 2):
            print('*', end=' ')
        print('\n')
                    
    def __getitem__(self, idx):
        return self.board[idx]

    