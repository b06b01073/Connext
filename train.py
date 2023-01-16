from connext import ConnextAgent
from replay_buffer import ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent, RandomAgent
from tqdm import tqdm
from copy import deepcopy
import torch
import matplotlib.pyplot as plt

def main():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()
    win_rates = []
    num_self_play = 30

    for i in tqdm(range(300), desc='Episode'):
        generate_dataset(connextAgent, replay_buffer, num_self_play)
        train(connextAgent, replay_buffer, 200)

        win_rate = bench_mark(connextAgent)
        win_rates.append(win_rate)
        plt.clf()
        plt.plot(win_rates)

        plt.savefig('win_rates.png')

        # print(f'episode: {i}, total loss: {total_loss}')
        if i % 5 == 0:
            torch.save(connextAgent.connext_net.state_dict(), f'model/model_params_{i}.pth')

def generate_dataset(connextAgent, replay_buffer, num_self_play):
    ''' Generate dataset that is going to be used in the next training step in main, and append the game history to the replay buffer

    Arguments: 
        connextAgent: the connext agent
        replay_buffer: the replay buffer
        num_self_play: the number of times the self-play is going to happen to generate the dataset, it is going to generate `num_self_play` of gameplay
    '''
    game_len = 0
    dataset = []
    for _ in tqdm(range(num_self_play), desc='Generating Dataset'):
        connextAgent.clean_history()
        board = Board()
        connextAgent.token = 1
        while not board.terminated:
            game_len += 1
            action = connextAgent.step(deepcopy(board))
            board.step(action, connextAgent.token)
            connextAgent.token = flip_token(connextAgent.token)

        winner_token = board.winner_token
        connextAgent.update_history(winner_token)
        dataset += connextAgent.history

    replay_buffer.append_dataset(dataset)
    print(f'avg game len: {game_len / num_self_play}')

def train(connextAgent, replay_buffer, epochs):
    total_mse = 0
    total_cross = 0
    for _ in range(epochs):
        mse_loss, cross_entropy_los = connextAgent.learn(replay_buffer)

        with torch.no_grad():
            total_mse += mse_loss
            total_cross += cross_entropy_los
    
    print(f'avg mse loss: {total_mse / epochs}, avg cross loss: {total_cross / epochs}')

def bench_mark(connextAgent, total_games=10):
    print(f'bench_marking...')
    with torch.no_grad():
        win = 0
        for i in range(total_games):
            env = ConnectX()
            env.embedded_player = RandomAgent()

            agent_token, board = env.register(connextAgent)
            connextAgent.token = agent_token

            while True:
                action = connextAgent.step(deepcopy(board))
                board, result, terminated = env.step(action)
                if terminated:
                    if result == 1:
                        win += 1
                    print(f'game {i}, result: {result}')
                    # env.render()
                    
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
    main()