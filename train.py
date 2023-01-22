from connext import ConnextAgent
from replay_buffer import ReplayBuffer
from env import Board, ConnectX
from agent import MCTSAgent, RandomAgent
from tqdm import tqdm
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def main():
    connextAgent = ConnextAgent()
    replay_buffer = ReplayBuffer()
    win_rates = []

    for i in tqdm(range(300), desc='Episode'):
        generate_dataset(connextAgent, replay_buffer)
        train(connextAgent, replay_buffer, epochs=400)


        bench_mark_games = 20
        win_games = 0
        with ThreadPoolExecutor(max_workers=bench_mark_games) as executor:
            futures = [executor.submit(bench_mark, deepcopy(connextAgent)) for _ in range(bench_mark_games)]
            for idx, future in enumerate(futures):
                result, game_len = future.result()
                if result == 1:
                    win_games += 1
                print(f'collected {idx}: {result}, game len: {game_len}')

        win_rate = win_games / bench_mark_games
        print(f'Win rate: {win_rate}')

        win_rates.append(win_rate)
        plt.clf()
        plt.ylim(0, 1)
        plt.plot(win_rates)

        plt.savefig('noisy_win_rates.png')

        torch.save(connextAgent.connext_net.state_dict(), f'model/noisy/model_params_{i}.pth')

def generate_dataset(connextAgent, replay_buffer):
    ''' Generate dataset that is going to be used in the next training step in main, and append the game history to the replay buffer

    Arguments: 
        connextAgent: the connext agent
        replay_buffer: the replay buffer
        num_self_play: the number of times the self-play is going to happen to generate the dataset, it is going to generate `num_self_play` of gameplay
    '''

    dataset = []
    connextAgent.clean_history()
    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(play, deepcopy(connextAgent)) for _ in range(max_workers)]
        for idx, future in enumerate(futures):
            result = future.result()
            dataset += result
            print(f'collect {idx}: {len(result)}')

    replay_buffer.append_dataset(dataset)
    print(f'collected board positions: {len(replay_buffer[-1])}')


def play(connextAgent):
    num_self_play = 5
    dataset = []
    for _ in range(num_self_play):
        connextAgent.clean_history()
        board = Board()
        connextAgent.token = 1
        while not board.terminated:
            action = connextAgent.step(deepcopy(board))
            board.step(action, connextAgent.token)
            connextAgent.token = flip_token(connextAgent.token)

        # flip the token back to get the token of the last player
        last_player_token = flip_token(connextAgent.token)

        # get the result from the last player's point of view
        result = get_result(last_player_token, board.winner_token)

        connextAgent.update_history(result)

        dataset += connextAgent.history

    return dataset

def get_result(last_player_token, winner_token):
    if winner_token == 0:
        return 0
    elif last_player_token == winner_token:
        return 1

def train(connextAgent, replay_buffer, epochs):
    total_mse = 0
    total_cross = 0
    for _ in range(epochs):
        mse_loss, cross_entropy_los = connextAgent.learn(replay_buffer)

        with torch.no_grad():
            total_mse += mse_loss
            total_cross += cross_entropy_los
    
    print(f'avg mse loss: {total_mse / epochs}, avg cross loss: {total_cross / epochs}')

def bench_mark(connextAgent):
    with torch.no_grad():
        connextAgent.training = False
        game_len = 0
        env = ConnectX()
        env.embedded_player = MCTSAgent(simulations=200)

        agent_token, board = env.register(connextAgent)
        connextAgent.token = agent_token

        while True:
            game_len += 1
            action = connextAgent.step(deepcopy(board))
            board, result, terminated = env.step(action)
            if terminated:
                return result, game_len


def flip_token(token):
    if token == 1:
        return 2
    else:
        return 1


if __name__ == '__main__':
    main()