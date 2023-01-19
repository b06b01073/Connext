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
    connext_agent = ConnextAgent(pre_load='model/model_params_70.pth', training=False)
    connext_agent.set_simulations(50)
    env.embedded_player = connext_agent
    

    agent = MCTSAgent(50)
    agent_token, board = env.register(agent)
    agent.token = agent_token


    while True:
        env.render()
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


    for i in tqdm(range(60)):
        run()
    print(f'win: {win}, loss: {loss}, draw: {draw}')