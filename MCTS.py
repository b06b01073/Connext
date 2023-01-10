import numpy as np
import math
from mcts_config import config 
from copy import deepcopy

C = config['ucb_c']

class Node:
    ''' A data structure represents a node in the Monte Carlo search tree(MCST).

    Public Attributes:
        state: a Board class instance that represents the state of the node
        is_leaf: a boolean that is True if the node is a leaf node
        is_root: a boolean that is True if the node is the root of a MCST
        children: a list of Node instances that represents the children of this node
        parent: a Node class instance that represents the parent of this node
        token: an integer in {1, 2} that represents which player to move given the the state
        is_terminal: a boolean that is True if the state is a terminal state
        visited_count: an integer that represents the number of times this node has been visited during MCTS
        expected_reward: a float in [-1, 1] that represents the expected reward from this node
        move_taken: an integer that represents the last move.
    '''
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
        ''' Get the children of this node

        Get the children of this node by first reading the legal moves, and creates the Node class instances for those legal moves.
        '''
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


def naive_mcts(board, token, simulations, rollout_policy):
    ''' Run a Monte Carlo tree search(mcts)

    Arguments: 
        board: a Board class instance 
        token: an integer that represents the token of the agent who makes this function call
        simulations: an integer that represent the number of simulations are going to be ran in the mcts.
        rollout_policy: a function implemented by the caller Agent instance which represents the rollout policy

    Returns:
        An integer that represents the move decided by the mcts.
    '''
    mcst = Node(board, token, is_root=True)

    # run 100 simulations 應該要以一顆mcst為單位做平行搜索，
    for simulation_count in range(simulations):
        # search until meet a leaf
        leaf = search_leaf(mcst, simulation_count)
        leaf = expand(leaf)

        # get the terminal node by following the rollout policy
        winner_token = rollout(leaf, rollout_policy)
        
        backpropagate(leaf, winner_token, mcst.token)


    return make_move(mcst)

def expand(leaf):
    ''' Expands a node by adding its children to it before the rollout stage

    Arguments:
        leaf: a Node class instance that represents the node that is going to be expanded

    Returns: 
        A Node class instance that represents a randomly choosed children of the leaf parameter.
    '''

    # cannot expand anymore
    if leaf.is_terminal:
        return leaf


    leaf.get_children()
    new_leaf = np.random.choice(leaf.children)
    new_leaf.visited_count = 1
    return new_leaf


def search_leaf(root, simulation_count):
    ''' This function call Implements the selection stage of mcts, it runs a sequence of selections until it meets a leaf node.

    Arguments:
        root: a Node class instance that represents the root of the mcst.
        simulation_count: an integer that represents the number of simulations has been ran.

    Returns:
        A Node class instance that represents the leaf node.  
    '''
    node = root 

    # loop until meet a leaf node(note that the leaf node can also be a terminal node)
    while not node.is_leaf:
        node.visited_count += 1     # increment visit counter
        node = ucb_search(node, simulation_count)

    node.visited_count += 1

    # note the returned node can be a terminal node
    return node

def ucb_search(node, simulation_count):
    ''' This ucb_search function supports the selection of search_leaf, it select the next visited node by UCB value

    Arguments: 
        node: a Node class instance that represents the parent node
        simulation_count: a integer that represents the number of simulations has been ran.

    Returns:
        A Node class instance that represents the children of the node parameter which is going to be visited next. 
    '''
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
    ''' Calculate the UCB value based on the expected_reward, simulation_count and visited_count

    Arguments:
        node: a Node class instance that represents the evaluated node
        simulation_count: an integer that represents the number of simulations has been ran.
    
    Returns:
        A float that represents the value of UCB result.
    '''
    return node.expected_reward + C * math.sqrt(math.log(simulation_count) / node.visited_count)

def rollout(node, rollout_policy):
    ''' Do the rollout from the node

    Arguments:
        node: a Node class instance that represents the starting point of the rollout
        rollout_policy: a function implemented by the caller Agent instance which represents the rollout policy

    Return:
        An integer that represents the token of the winner after the rollout. 
    '''

    # do not rollout an terminal state
    if node.is_terminal:
        return node.state.winner_token

    # keep playing the game until it end
    winner_token = rollout_policy(node)

    return winner_token



def backpropagate(node, winner_token, root_token):
    ''' Updates the rollout result all the way up to root

    Updates the expected_reward attribute of the nodes from this node all the way up to root.

    Arguments:
        node: a Node class instance that represents the rolled out leaf node in the mcst
        winner_token: an integer that represents the winner of the rollout
        root_token: an integer that represents the token attribute of the root node in the mcst 
    '''

    # backpropagate all the way up to root
    while not node.is_root:
        score = get_score(winner_token, root_token) # ex: if winner_token == node.token, then the current player is expected to win from this state
        node.expected_reward = update_expected_reward(node, score)
        node = node.parent


def get_score(winner_token, root_token):
    ''' Decides the score of a simulation that is going to take part in the update.

    Arguments:
        winner_token: an integer that represents the token attribute of the root node in the mcst 
        root_token: an integer that represents the token attribute of the root node in the mcst 

    Returns:
        An integer that represents the result of a simulation, the returned value is defined as follow:
            1: the player won the game
            0: the game is a draw
            -1: the player lost the game 

    '''

    score = None
    if winner_token == root_token:
        score = 1
    elif winner_token == 0:
        score = 0
    else:
        score = -1
    return score


def update_expected_reward(node, score):
    ''' Updated the expected_reward of a Node class instance.

    Attributes: 
        node: A Node class instance that represents the updated node
        score: An integer that represents the result of a simulation

    Returns: 
        The updated expected_reward of the node parameter.
    '''
    return (node.expected_reward * (node.visited_count - 1) + score) / node.visited_count

def make_move(mcst):
    ''' Suggests a move based on MCTS

    This function suggest a move based on the result of MCTS. It picks the most visited children as the next move.

    Arguments:
        mcts: a Node class instance that represents the root of the MCST

    Returns:
        An integer that represents suggested move.
    '''

    visited_counts = np.array([child.visited_count for child in mcst.children])
        
    best_move = np.argsort(visited_counts)[-1]

    return mcst.children[best_move].move_taken