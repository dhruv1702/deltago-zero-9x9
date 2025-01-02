from typing import Tuple, Dict

import torch

from game_mechanics import is_terminal
from go_base import all_legal_moves
from state import State
from utils import WHITE, BOARD_SIZE, PASS_MOVE, PlayerMove

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device =  torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


NodeID = Tuple[PlayerMove, ...]


class Node:
    def __init__(self, state: State):
        self.state = state
        self.is_terminal = is_terminal(state)

        # Make legal moves mask
        self.legal_moves_mask = torch.zeros((1, BOARD_SIZE ** 2 + 1,), device=device)
        self.legal_moves = torch.from_numpy(all_legal_moves(self.state.board, self.state.ko))
        self.legal_moves_mask[0, self.legal_moves] = 1

        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_node_ids = self._get_possible_children()
        self.key: NodeID = self.state.recent_moves

        # We suggest you cache these to speed up runtime, although this is optional :)
        self.cnn_input = to_cnn_input(self.state)
        self.fc_input = to_fc_input(self.state)

    def _get_possible_children(self) -> Dict[int, NodeID]:
        """Gets the possible children of this node."""
        if self.is_terminal:
            return {}
        # Note: possibility for a speedup here by not getting the children until
        #  they are chosen. Would require changing state id to prev state id + action
        #  so we can lookup without having to recompute the children.
        # This only took up ~15% of the time in the profiler, so it's not a huge deal.
        return {
            int(action): self.state.recent_moves + (PlayerMove(self.state.to_play, int(action)),)
            for action in self.legal_moves
        }


NUM_INPUT_CHANNELS = 4


def to_cnn_input(state: State) -> torch.Tensor:
    #nn_input = torch.zeros((1, NUM_INPUT_CHANNELS, 9, 9), device=device)
    #nn_input = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(state.board),0),1).float().to(device)
    # Set the elements based on the state
    player_color_channel = torch.full((9, 9), state.player_color)
    state_channel = torch.from_numpy(state.board)
    black_channel = (state_channel==1).int()
    white_channel = (state_channel==-1).int()
    nn_input = torch.unsqueeze(torch.stack((player_color_channel,state_channel,black_channel,white_channel)),0).float()
    #print(nn_input.shape)
    #print(nn_input.dtype)
    return nn_input.to(device=device)


def to_fc_input(state: State) -> torch.Tensor:
    nn_input = torch.zeros((1, 6), device=device)
    # Set the elements based on the state
    ...
    return nn_input