from env import ConnectX
from connext import ConnextAgent
from agent import Human

if __name__ == '__main__':
    model_path = 'model/noisy/model_params_70.pth'
    env = ConnectX()
    env.embedded_player = ConnextAgent(pre_load=model_path)
    env.embedded_player.set_simulations(800)
    
    human = Human()
    _, board = env.register(human)
    env.render()

    while True:
        action = human.step(board)
        board, result, terminated = env.step(action)
        env.render()
        if terminated:
            if result == 1:
                print('you won')
            elif result == 0:
                print('draw')
            else:
                print('you lose')
            break