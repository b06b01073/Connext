from connext import ConnextAgent
from replay_buffer import ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent, RandomAgent
from tqdm import tqdm
from copy import deepcopy
import torch
import matplotlib.pyplot as plt

def train():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()
    win_rates = []

    for i in tqdm(range(2500)):
        board = Board()
        connextAgent.token = 1
        connextAgent.clean_history()

        while not board.terminated:
            action = connextAgent.step(deepcopy(board))
            board.step(action, connextAgent.token)

            connextAgent.token = flip_token(connextAgent.token)
            connextAgent.learn(replay_buffer)

        winner_token = board.winner_token
        connextAgent.update_history(winner_token)
        replay_buffer.append_history(connextAgent.history)


        if (i + 1) % 50 == 0:
            win_rate = bench_mark(connextAgent)
            win_rates.append(win_rate)
            plt.clf()
            plt.plot(win_rates)

        plt.savefig('win_rates.png')

        # print(f'episode: {i}, total loss: {total_loss}')


def bench_mark(connextAgent, total_games=10):
    print(f'bench_marking...')
    with torch.no_grad():
        win = 0
        for i in range(total_games):
            env = ConnectX()
            env.embedded_player = RandomAgent()

            agent = connextAgent
            agent_token, board = env.register(agent)
            agent.token = agent_token

            while True:
                action = agent.step(deepcopy(board))
                board, result, terminated = env.step(action)
                if terminated:
                    if result == 1:
                        win += 1
                    print(f'game {i}, result: {result}')
                    env.render()
                    
                    break



        win_rate = win / total_games
        print(f'win rate: {win_rate}')
        return win_rate
            

def flip_token(token):
    if token == 1:
        return 2
    else:
        return 1


if __name__ == '__main__':
    train()