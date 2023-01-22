from agent import RandomAgent, Human, MCTSAgent
from env import ConnectX
from tqdm import tqdm
from connext import ConnextAgent
from copy import deepcopy

win = 0
lose = 0
draw = 0

def run():
    global win
    global lose
    global draw

    model_path = 'model/model_params.pth'

    env = ConnectX()
    embedded_agent = MCTSAgent(4000)
    env.embedded_player = embedded_agent

    connext_agent = ConnextAgent(pre_load=model_path, training=False)
    connext_agent.set_simulations(200)
    

    agent_token, board = env.register(connext_agent)
    connext_agent.token = agent_token


    while True:
        env.render()
        action = connext_agent.step(deepcopy(board))
        board, result, terminated = env.step(action)

        if terminated:
            if result == 1:
                win += 1
            elif result == 0:
                draw += 1
            else:
                lose += 1
            break


if __name__ == '__main__':

    games = 60
    for i in tqdm(range(games)):
        run()
    print(f'win: {win}, lose: {lose}, draw: {draw}')