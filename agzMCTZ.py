import math
import torch
import random
import numpy as np

from typing import Dict, Tuple, List, Optional
from torch import nn
from tqdm import tqdm
from go_base import is_move_legal,all_legal_moves
from game_mechanics import (
    transition_function,
    is_terminal,
    choose_move_randomly,
    play_go,
    reward_function, save_pkl,
)
from node import Node, NodeID
from utils import BLACK, BOARD_SIZE,int_to_coord
from state import State

device =  torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#torch.set_default_dtype(torch.float32)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_NUM_MOVES = BOARD_SIZE**2 + 1

my_max = max

class AGZMCTS:
    def __init__(
        self,
        initial_state: State,
        c_puct: float,
        network,
        eta_noise_root_node: float = 0.25,
        verbose: int = 0,
    ):
        self.root_node = Node(initial_state)
        self.eta_noise_root_node = eta_noise_root_node
        # Maps node IDs to nodes
        self.tree: Dict[str:Node] = {self.root_node.key: self.root_node}
        self.total_return: Dict[str:float] = {self.root_node.key: 0.0}
        self.N: Dict[str:int] = {self.root_node.key: 0}

        # Get the neural network policy at the root node for use in PUCT
        v, pol = network(self.root_node.cnn_input,self.root_node.legal_moves_mask)

        # if tree is pruned then dont add noise to root here
        #pol_with_noise = (1 - eta_noise_root_node) * pol.squeeze().detach().numpy() + \
         #           eta_noise_root_node * np.random.dirichlet(np.full(pol.squeeze().shape[0], 0.03))
        pol_with_noise = self.add_dirichlet_noise(pol)

        self.pol: Dict[str:torch.tensor] = {self.root_node.key:pol_with_noise}
        self.c_puct = c_puct
        self.network = network

        self.verbose = verbose
    
    def add_dirichlet_noise(self, probs, alpha=0.03):
        pol_with_noise = (1 - self.eta_noise_root_node) * probs.squeeze().detach().cpu().numpy() + \
                    self.eta_noise_root_node * np.random.dirichlet(np.full(probs.squeeze().detach().shape[0], alpha))
        return torch.from_numpy(pol_with_noise.astype('float32')).to(device)

    def do_rollout(self) -> None:
        if self.verbose:
            print("\nNew rollout started from", self.root_node.key)
        path_taken = self._select()
        chosen_node = self._expand(path_taken[-1])
        value = self._evaluate(chosen_node)
        self._backup(path_taken, chosen_node, value)

    def _select(self) -> List[NodeID]:
        """Selects a node to simulate from, given the current state and tree.

        Returns a list of nodes of the path taken from the root
         to the selected node.
        """
        node = self.root_node
        if self.verbose:
            print("Selecting from node:", node.key)
        path_taken = [node.key]

        while not node.is_terminal and node.key in self.tree:
            node_id = self._alphago_select(node)
            path_taken.append(node_id)

            if node_id not in self.tree:
                if self.verbose:
                    print("Node not in tree, expanding:", node_id)
                break
            # Update `node` object for next iteration
            node = self.tree[node_id]

        return path_taken

    def _expand(self, node_id: NodeID) -> Node:
        """Unless the selected node is a terminal state, expand the selected node by adding its
        children nodes to the tree.
        """
        if node_id in self.tree:
            return self.tree[node_id]

        if self.verbose:
            print("Expanding node:", node_id)

        parent_node = self.tree[node_id[:-1]]
        # Action taken to get to this node is the last element of the node ID
        new_state = transition_function(parent_node.state, node_id[-1].move)
        child_node = Node(new_state)
        self.tree[child_node.key] = child_node
        self.total_return[child_node.key] = 0
        self.N[child_node.key] = 0
        return child_node

    def _evaluate(self, node: Node) -> float:
        """
        Evaluates a node

        Want to use:
        ```
        value, policy = self.network(node)
        ```
        to get the value of `node`.

        Treat terminal states differently - use the reward function.

        Cache policy output so you can use it in future PUCT calculations.
        """
        ...
        if node.is_terminal:
            if node.state.player_color==1:
                reward = reward_function(node.state)
            elif node.state.player_color==-1:
                reward = -1 * reward_function(node.state)
            return reward
        else:
            value, policy = self.network(node.cnn_input,node.legal_moves_mask)
            self.pol[node.key] = policy
            return value


    def _backup(self, path_taken: List[NodeID], evaluated_node: Node, backup_value: float) -> None:
        """
        Update the action-value estimates of all parent nodes in the tree with the
        return from the simulated trajectory.

        Remember to consider whether values are relative to the current player or not.
         This is a design choice you make - what are you training your neural network
         to predict?
        """
        for node_id in path_taken:
            # Since values are in range [-1, 1], inverse is -value
            node = self.tree[node_id]
            node_backup_value = backup_value if node.state.to_play == evaluated_node.state.to_play else -1 * backup_value
            self.total_return[node_id] += node_backup_value
            self.N[node_id] += 1

            if self.verbose >= 2:
                print(
                    "Backing up node:",
                    node_id,
                    self.N[node_id],
                    self.total_return[node_id],
                )

    def choose_action(self, temperature: float) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Once we've simulated all the trajectories, we want to
         select the action at the current timestep which
         maximises the action-value estimate.
        """
        if self.verbose:
            print("N:", {a: self.N.get(state, 0) for a, state in self.root_node.child_node_ids.items()})

        if temperature == 0:
            # Use most-visited - do this when evaluating the agent
            return my_max(
                self.root_node.child_node_ids.keys(),
                key=lambda a: self.N.get(self.root_node.child_node_ids[a],0),
            ), None

        # Fill in a lot here to calculate the action probabilities and sample from it
        counts = []     # visits to each child node
        actions = [] 
        for a, node_id in self.root_node.child_node_ids.items():
            if is_move_legal(int_to_coord(a),self.root_node.state.board,self.root_node.state.ko):
                counts.append(self.N.get(node_id,0))
                actions.append(a)
        counts = [x**(1./temperature) for x in counts]

        child_node_probs = [x/float(sum(counts)) for x in counts]

        output_probs = np.zeros((9 ** 2 + 1,),dtype=np.float32)
        for i, action in enumerate(actions):
            output_probs[action] = child_node_probs[i]

        chosen_action = np.random.choice(range(len(output_probs)), p=output_probs)
        if self.verbose:
            print("Action probabilities:", output_probs)
            print("Chosen action:", chosen_action)
        return chosen_action, torch.from_numpy(output_probs)


    def Q(self, node_id: Tuple) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    
    def _alphago_select(self, parent: Node) -> NodeID:
        """Implement PUCT algorithm from AlphaGo Zero paper."""
        #raise NotImplementedError()
        n_parent = self.N[parent.key]
        child_nodes = parent.child_node_ids         # child_nodes is a dict: {action:node_key}
        action_scores = {}
        for action in child_nodes.keys():
            if is_move_legal(int_to_coord(action),parent.state.board,parent.state.ko):
                if (child_nodes[action] not in self.tree):
                    Qval = self.Q(parent.key)
                else:
                    Qval = -self.Q(child_nodes[action])
                action_scores[action] = Qval + ((self.c_puct * self.pol[parent.key].squeeze()[action]* \
                    np.sqrt(n_parent)) / (1 + self.N.get(child_nodes[action],0)))
        
        # max_action = max(action_scores, key=action_scores.get)  # todo: if multiple actions have equal max- np.random.choice()
        m = max(action_scores.values())
        max_action = random.choice([k for k in action_scores if action_scores[k] == m])
        #max_acts = [x for x in action_scores.keys() if action_scores[x] == action_scores[max(action_scores, key=action_scores.get)]]
        #max_action = random.choice(max_acts)
        return child_nodes[max_action]


    def prune_tree(self, new_root_state: State) -> None:
        """This is optional, but will lead to an improvement in performance.
        You can prune the tree to only include nodes which are relevant to the new state.
         I.e. only include all ancestors of the new root node.
        """
        # If it's the terminal state we don't care about pruning the tree
        if is_terminal(new_root_state):
            return

        self.root_node = Node(new_root_state)

        self.N[self.root_node.key] = 0
        self.total_return[self.root_node.key] = 0

        # Build a new tree dictionary
        new_tree = {self.root_node.key: self.root_node}

        prev_added_nodes = {self.root_node.key: self.root_node}
        while prev_added_nodes:
            newly_added_nodes = {}

            for node in prev_added_nodes.values():
                child_nodes = {
                    node.key: self.tree[node_id]
                    for node_id in node.child_node_ids.values()
                    if node.key in self.tree
                }
                new_tree.update(child_nodes)
                newly_added_nodes.update(child_nodes)

            prev_added_nodes = newly_added_nodes

        self.tree = new_tree
        self.total_return = {key: self.total_return[key] for key in self.tree}
        self.N = {key: self.N[key] for key in self.tree}

