from connext import ConnextAgent, ReplayBuffer
from env import Board

def train():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()

    for i in range(10):
        board = Board()
        connextAgent.token = 1

        while not board.terminated:
            action = connextAgent.step(board)
            board.step(action, connextAgent.token)

            # connextAgent.learn()
            connextAgent.token = flip_token(connextAgent.token)


            

def flip_token(token):
    if token == 1:
        return 2
    else:
        return 1


if __name__ == '__main__':
    train()