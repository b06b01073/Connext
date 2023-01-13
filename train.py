from connext import ConnextAgent
from replay_buffer import ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent
from tqdm import tqdm
from copy import deepcopy
import torch

def train():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()

    for i in tqdm(range(200)):
        board = Board()
        connextAgent.token = 1

        while not board.terminated:
            action = connextAgent.step(deepcopy(board))
            board.step(action, connextAgent.token)

            connextAgent.token = flip_token(connextAgent.token)
            connextAgent.learn(replay_buffer)

        winner_token = board.winner_token
        connextAgent.update_history(winner_token)

        replay_buffer.append_history(connextAgent.history)
        connextAgent.clean_history()

        # if i % 5 == 0:
        #     bench_mark(connextAgent)

        # print(f'episode: {i}, total loss: {total_loss}')


def bench_mark(connextAgent, total_games=15):
    print(f'bench_marking...')
    with torch.no_grad():
        win = 0
        for _ in tqdm(range(total_games)):
            env = ConnectX()
            env.embedded_player = MCTSAgent(simulations=500)

            agent = connextAgent
            agent_token, board = env.register(agent)
            agent.token = agent_token

            while True:
                action = agent.step(board)
                board, result, terminated = env.step(action)
                if terminated:
                    if result == 1:
                        win += 1
                    break
        print(f'win rate: {win / total_games}')

            

            

def flip_token(token):
    if token == 1:
        return 2
    else:
        return 1


if __name__ == '__main__':
    train()