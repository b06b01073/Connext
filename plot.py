import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open('result.txt') as f:
        lines = f.readlines()
        
        agent = []
        wins = []
        loses = []
        draws = []

        for line in lines:
            line = line.strip('\n').split(', ')
            print(line)
            agent.append(line[0])
            wins.append(int(line[1]))
            loses.append(int(line[2]))
            draws.append(int(line[3]))

        x = np.arange(len(agent))  # the label locations
        width = 0.2  # the width of the bars

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.bar(x - width, wins, width, label='Win')
        plt.bar(x, loses, width, label='Lose')
        plt.bar(x + width, draws, width, label='Draw')
        plt.ylabel('Count')
        plt.title('Result of Connext200')
        plt.xticks(x, agent)
        plt.legend()



        plt.savefig('plot.png')