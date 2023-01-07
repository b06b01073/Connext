from agent import RandomAgent, Human
from env import ConnectX

env = ConnectX()
env.embedded_player = RandomAgent()

agent = RandomAgent()
agent_token, board = env.register(agent)

while True:
    action = agent.step(board)
    board, result = env.step(action)
    env.render()
    if result != 0:
        break



