from agent import RandomAgent, Human, MCTSAgent
from env import ConnectX
from tqdm import tqdm

win = 0
loss = 0
draw = 0

def run():
    global win
    global loss
    global draw

    env = ConnectX()
    env.embedded_player = MCTSAgent(simulations=800000)

    agent = Human()
    agent_token, board = env.register(agent)
    agent.token = agent_token

    while True:
        action = agent.step(board)
        board, result = env.step(action)
        if result != 0:
            if result == 1:
                win += 1
            elif result == 0:
                draw += 1
            else:
                loss += 1
            break
        env.render()

if __name__ == '__main__':
    for i in tqdm(range(1)):
        run()
    print(f'win: {win}, loss: {loss}, draw: {draw}')