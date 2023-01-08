import numpy as np
import math
from mcts_config import config 
from copy import deepcopy

C = config['ucb_c']

class Node:
    def __init__(self, board, token, move_taken=None, parent=None, is_root=False):
        self.state = board
        self.is_leaf = True
        self.is_root = is_root
        self.children = [] # a list of children Node, we do a lazy update here.
        self.parent = parent # record the parent for backpropagation
        self.token = token # if to_play == k, then it's player with token k to play given the self.state(k is in {1, 2}).
        self.is_terminal = board.terminated # the node is a terminal state if the game ends at this node(we can check this condition by reading Board.terminated attribute)
        self.visited_count = 0
        self.expected_reward = 0
        self.move_taken = move_taken # which move does the previous state taken so that previous state transits to this current state

    def get_children(self):
        legal_moves = self.state.get_legal_moves()
        # get the token of the next player
        next_token = 1 if self.token == 2 else 2

        for legal_move in legal_moves:
            # copy the board and play the legal move
            copied_board = deepcopy(self.state)
            copied_board.step(legal_move, self.token)

            child = Node(board=copied_board, token=next_token, move_taken=legal_move, parent=self)

            self.children.append(child)
            self.is_leaf = False


def naive_mcts(board, token, simulations):
    '''
    This function call takes a board position as a root of Monte Carlo search tree, and follows the `rollout_policy` during the rollout stage.
    '''
    mcst = Node(board, token, is_root=True)

    # run 100 simulations 應該要以一顆mcst為單位做平行搜索，
    for simulation_count in range(simulations):
        # search until meet a leaf
        leaf = search_leaf(mcst, simulation_count)
        leaf = expand(leaf)

        # get the terminal node by following the rollout policy
        winner_token = rollout(leaf)
        
        backpropagate(leaf, winner_token, mcst.token)


    return make_move(mcst)

def expand(leaf):
    # cannot expand anymore
    if leaf.is_terminal:
        return leaf


    leaf.get_children()
    new_leaf = np.random.choice(leaf.children)
    new_leaf.visited_count = 1
    return new_leaf


def search_leaf(root, simulation_count):
    node = root 

    # loop until meet a leaf node(note that the leaf node can also be a terminal node)
    while not node.is_leaf:
        node.visited_count += 1     # increment visit counter
        node = ucb_search(node, simulation_count)

    node.visited_count += 1

    # note the returned node can be a terminal node
    return node

def ucb_search(node, simulation_count):
    max_ucb = float('-inf')
    res = 0

    for i in range(len(node.children)):
        if node.children[i].visited_count == 0:
            return node.children[i]
        ucb_value = get_ucb(node.children[i], simulation_count)
        if ucb_value > max_ucb:
            max_ucb = ucb_value
            res = i

    return node.children[res]


def get_ucb(node, simulation_count):
    return node.expected_reward + C * math.sqrt(math.log(simulation_count) / node.visited_count)

def rollout(node):

    # do not rollout an terminal state
    if node.is_terminal:
        return node.state.winner_token

    # keep playing the game until it end
    winner_token = rollout_policy(node)

    return winner_token

def rollout_policy(node):
    board = deepcopy(node.state)
    token = node.token
    while True:
        legal_moves = board.get_legal_moves()

        move = np.random.choice(legal_moves)
        board.step(move, token)

        if board.terminated:
            break

        if token == 1:
            token = 2
        else:
            token = 1
        
    return board.winner_token

def backpropagate(node, winner_token, root_token):
    # backpropagate all the way up to root
    while not node.is_root:
        score = get_score(winner_token, root_token) # ex: if winner_token == node.token, then the current player is expected to win from this state
        node.expected_reward = update_expected_reward(node, score)
        node = node.parent


def get_score(winner_token, root_token):
    score = None
    if winner_token == root_token:
        score = 1
    elif winner_token == 0:
        score = 0
    else:
        score = -1
    return score


def update_expected_reward(node, score):
    return (node.expected_reward * (node.visited_count - 1) + score) / node.visited_count

def make_move(mcst):
    visited_counts = np.array([child.visited_count for child in mcst.children])
        
    best_move = np.argsort(visited_counts)[-1]

    return mcst.children[best_move].move_taken