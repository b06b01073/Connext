from agent import RandomAgent, Human, MCTSAgent
from env import ConnectX
from tqdm import tqdm
from connext import ConnextAgent
from copy import deepcopy

win = 0
loss = 0
draw = 0

def run():
    global win
    global loss
    global draw

    env = ConnectX()
    env.embedded_player = RandomAgent()

    agent = MCTSAgent(simulations=50)
    agent_token, board = env.register(agent)
    agent.token = agent_token


    while True:
        action = agent.step(deepcopy(board))
        board, result, terminated = env.step(action)


        if terminated:
            if result == 1:
                win += 1
            elif result == 0:
                draw += 1
            else:
                loss += 1

            break


if __name__ == '__main__':


    for i in tqdm(range(40)):
        run()
    print(f'win: {win}, loss: {loss}, draw: {draw}')