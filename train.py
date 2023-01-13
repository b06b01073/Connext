from connext import ConnextAgent
from replay_buffer import ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent
from tqdm import tqdm
from copy import deepcopy

def train():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()

    for i in tqdm(range(7)):
        board = Board()
        connextAgent.token = 1

        while not board.terminated:
            action = connextAgent.step(deepcopy(board))
            board.step(action, connextAgent.token)

            # connextAgent.learn()
            connextAgent.token = flip_token(connextAgent.token)

            connextAgent.learn(replay_buffer)

        winner_token = board.winner_token
        connextAgent.update_history(winner_token)

        replay_buffer.append_history(connextAgent.history)
        connextAgent.clean_history()

        


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