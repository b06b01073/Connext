from connext import ConnextAgent, ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent
from tqdm import tqdm

def train():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()

    for i in tqdm(range(1)):
        board = Board()
        connextAgent.token = 1

        while not board.terminated:
            action = connextAgent.step(board)
            board.step(action, connextAgent.token)

            # connextAgent.learn()
            connextAgent.token = flip_token(connextAgent.token)

        if i % 10 == 0:
            bench_mark(connextAgent)


def bench_mark(connextAgent):
    env = ConnectX()
    env.embedded_player = MCTSAgent(simulations=5)

    agent = connextAgent
    agent_token, board = env.register(agent)
    agent.token = agent_token

    while True:
        action = agent.step(board)
        board, result, terminated = env.step(action)
        if terminated:
            break

            

def flip_token(token):
    if token == 1:
        return 2
    else:
        return 1


if __name__ == '__main__':
    train()