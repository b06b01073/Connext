import numpy as np
from env_config import config
from termcolor import cprint, colored

class ConnectX:
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
        '''
        The agent calls ConnectX.step(col) will put an token at the col-th column.

        Return value:
            1, if self.agent won
            -1, if self.embedded_player won
            0, otherwise
        '''
        self.last_agent_move = self.board.step(col, self.agent_token)
        if self.board.terminated:   # the self.agent won by making the move 
            self.terminated = True
            return self.board, 1

        col = self.embedded_player.step(self.board)
        self.last_embedded_player_move = self.board.step(col, self.embedded_player_token)
        if self.board.terminated:   # the self.embedded_player won by making the move
            self.terminated = True
            return self.board, -1
        
        # non-terminal state
        return self.board, 0
        

    def register(self, agent):
        '''
        The ConnectX.register(agent) call register the agent as self.agent.

        Once an agent registerd in the env, the env will decide who goes first by the self.pick_first() function call. The player goes first has token of 1, the player goes second has token of 2.

        This function call also return the token of self.agent and the first observation of the board.
        '''

        self.agent = agent

        self.pick_first()

        # the embedded_agent plays the first move
        if self.first == 0:
            col = self.embedded_player.step(self.board)
            self.last_embedded_player_move = self.board.step(col, self.embedded_player_token)
        
        self.display_start()

        return self.agent_token, self.board

    def pick_first(self):
        '''
        decide the first player
        '''
        self.first = np.random.randint(low=0, high=2)

        if self.first == 0:
            self.embedded_player_token = 1
            self.agent_token = 2
        else:
            self.embedded_player_token = 2
            self.agent_token = 1

    def display_start(self):
        '''
        Display the starting position from the agent's point of view. If the agent goes first, then the starting board should be empyt. If the embedded_player goes first, then the starting board should have one token.
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
        '''
        print the self.board
        '''
        self.board.render(self.last_embedded_player_move, self.last_agent_move, self.embedded_player_token, self.agent_token)

class Board:
    def __init__(self):
        self.height = config['height']
        self.width = config['width']
        self.X = config['X']
        self.board = np.zeros((self.height, self.width), dtype='uint8')
        self.terminated = False
        self.steps = 0

    def step(self, col, token):
        '''
        The ConnectX env put an token on the col-th column, and it also check whether the move ends the game. 
        '''
        assert col >= 0 and col < self.width and self.board[0][col] == 0, 'Illegal move!'

        for row in range(self.height):
            if row == self.height - 1 or self.board[row + 1][col] != 0:
                self.board[row][col] = token
                self.terminated = self.__is_end(row, col, token)
                self.steps += 1
                return row, col


    def __is_end(self, row, col, token):
        '''
        The __is_end uses coord (row, col) as a center to check the winning condition.

        The returned value `terminated` is decided based on two conditions:
            1. Whether there is a winner
            2. Whether the game is a draw        
        '''

        terminated = self.__check_horizontal(row, col, token) or self.__check_vertical(row, col, token) or self.__check_diagonal(row, col, token) or self.steps == self.width * self.height

        return terminated

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
        while col - i >= 0 and row + i < self.height:
            if self.board[row + i][col - i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

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

        # downright
        while col + i < self.width and row + i < self.height:
            if self.board[row + i][col + i] == token:
                connected_tokens += 1
            else:
                break
            i += 1

        return True if connected_tokens >= self.X else False
        

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