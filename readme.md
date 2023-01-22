# Connext: A Connect 4 Program Based on Deep Learning and Monte Carlo Tree Search

## Introduction

This is a side project inspired by DeepMind's research on AlphaGo, AlphaGo Zero and AlphaZero. This project is done by [b06b01073](https://github.com/b06b01073) and [wasd9813](https://github.com/wasd9813).

### Connect 4
According to [Wikipedia](https://en.wikipedia.org/wiki/Connect_Four): 
Connect Four (also known as Connect 4) is a two-player connection board game, in which the players choose a color and then take turns dropping colored tokens into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own tokens. 

### Connext
In this project, we built an agent called Connext that learns how to play Connect 4 without any human intervention or human labeled data, Connext knows only the rules of the game. This makes Connext a highly generalized model which should be able to learn to play other games by providing solely the rules, since it requires no hand-crafted data or guidance from human experts. 

During the training, it generates its own traning data from the self-play game records. From the result(win, lose or draw) of those games, it learns how to make better decisions.

### Implementation
In our implementaion, we combine the policy network and value network into one single network, just like what AlphaGo Zero did. While making decisions, Connext performs Monte Carlo tree search(MCTS) guided by the policy and value network without any rollout when the search reaches a leaf node.

We use ResNet as the backbone(5 residual blocks in total), and the output from ResNet is sent to the policy network and value network respectively.

## How to run this project

```
$ python train.py       # train the agent
$ python main.py        # compare the agent performance
$ python play.py        # play a game against the Connext program
$ python model.py       # get the torchsummary of the model
```

## Evaluation
In the evalution part, we use the agent that make decisions based purely on MCTS as the benchmark(we will use the notation <agent_name>X to represent the agent that runs X simulations per move. For example, MCTS3000 means the MCTS agent runs 3000 simulations per move). 

We run Connext200 against MCTS3000 to test the strength of our agent(the model parameters is in the `model` folder). They play 60 games head-to-head

There are some work can be done to improve the performance of the model:

* Add more residual blocks to the model in `model.py`. This should theoretically improve the performance of the model.
* Increase the number of simulations during training. This will generate high quality dataset.

The reason why we did not do the things mentioned above is due to the constraint from our hardware(RTX 2070). It takes roughly 50 hours of training for 100 iterations :(.

## References
1. [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
2. [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
3. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
](https://arxiv.org/abs/1712.01815)
4. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)