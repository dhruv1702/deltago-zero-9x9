import random
import torch
import numpy as np
import torch.nn.functional as F
import time
import pickle
import cProfile

from typing import Any, Dict, Optional
from tqdm import tqdm

from check_submission import check_submission
#from game_mechanics import choose_move_randomly, load_pkl, play_go, save_pkl, human_player
from state import State
from go_base import all_legal_moves
from network import AGZNetwork
from game_mechanics import (GoEnv,
    transition_function,
    is_terminal,
    human_player,
    choose_move_randomly,
    play_go,
    reward_function, save_pkl,load_pkl
)
from node import Node, NodeID
from utils import BLACK, BOARD_SIZE
from agzMCTZ import AGZMCTS
#torch.set_default_dtype(torch.float32)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device =  torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Training on device: ',device)

TEAM_NAME = "DeltaGoZero"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

TOTAL_NUM_MOVES = BOARD_SIZE**2 + 1

my_max = max


def train(network=None, memory=None):
    # Hyperparams
    lr = 0.01
    batch_size = 200
    num_episodes = 10
    c_puct = 1.5  # exploration hyperparam in PUCT
    c_value = 1.5  # weight of value loss
    tempThreshold = 20
    # KataGo hyperparams
    prob_full_search = 0.25
    full_search_cap = 150
    short_search_cap = 25

    network = AGZNetwork(15,128).to(device) if network==None else network
    memory = [] if memory==None else memory
    # if network ==None:
    #     network = AGZNetwork(15,128).to(device)
    # else:
    #     network = network

    # if memory == None:
    #     memory = []
    # else:
    #     memory = memory
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    for episode in tqdm(range(num_episodes), "Episodes"):
        print("Episode", episode)
        # Default player_color is BLACK
        state = State()
        # mcts = AGZMCTS(
        #     initial_state=state,
        #     c_puct=c_puct,
        #     network=network,
        #     verbose=0
        # )
        done = False
        game_memory = []
        while not done:
            # KataGo addition - if confused, simply remove this
            #  and set num_iterations manually
            mcts = AGZMCTS(
                initial_state=state,
                c_puct=c_puct,
                network=network,
                verbose=0
            )
            full_search = random.random() < prob_full_search
            num_iterations = full_search_cap if full_search else short_search_cap

            print(f"Running {num_iterations} MCTS rollouts\t {len(mcts.tree)} nodes in tree\t poss moves: {len(mcts.root_node.legal_moves)}")

            for _ in range(num_iterations):
                mcts.do_rollout()

            temperature = 1 if (episode < tempThreshold) else 1e-2      # temp=1 for first 20 episodes, then infinitesimal 
            action, mcts_probs = mcts.choose_action(temperature=temperature)
            state = transition_function(state, int(action))
            done = is_terminal(state)
            node = Node(state)
            if full_search:
                game_memory.append((node,mcts_probs)) # TODO

        # Reward function only nonzero at terminal state
        black_outcome = reward_function(state)  # Reward relative to player (BLACK)
        if state.player_color==1:
            winner = black_outcome
        elif state.player_color==-1:
            winner = -1 * black_outcome     # winner relative to player

        # Adding memory from most recent game to overall memory upon termination
        memory += [game_entry + (winner,) for game_entry in game_memory]
        print("Game over, num in memory:{}. Winner = {}".format(len(memory),winner))
        # This is training episodically
        if len(memory) >= batch_size:
            update_network(network, optimizer, memory, batch_size, c_value)
    
    return network.cpu(), memory


def update_network(network,optimizer,memory, batch_size,c_value):
    print("Updating network")
    batch = random.sample(memory,batch_size)
    cnn_inputs = torch.cat([x[0].cnn_input for x in batch]).to(device)
    legal_masks = torch.cat([x[0].legal_moves_mask for x in batch]).to(device)
    mct_probs = torch.stack([x[1] for x in batch]).to(device)
    winners =  torch.FloatTensor([x[2] for x in batch]).to(device)

    # compute output
    vals, pis = network(cnn_inputs,legal_masks)
 
    policy_loss = F.cross_entropy(pis, mct_probs)
    value_loss = F.mse_loss(vals.squeeze(), winners)
    loss = policy_loss + c_value * value_loss

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record loss
    print("total loss = ", loss)
    return



def choose_move(
    state: State,
    pkl_file: Optional[Any] = None,
    mcts: Optional[AGZMCTS] = None,
) -> int:
    """Called during competitive play.
     It returns a single action to play.

    Args:
        state: The current state of the go board (see state.py)
        pkl_file: The pickleable object you returned in train
        env: The current environment

    Returns:
        The action to take
    """
    start = time.time()
    if mcts is None:
        mcts = AGZMCTS(
            initial_state=state,
            c_puct=1.5,
            network=pkl_file,
            verbose=0
        )
    else:
        mcts.prune_tree(state)

    rollout_count = 0
    while True:
        mcts.do_rollout()
        rollout_count += 1
        elapsed = time.time() - start
        if elapsed > 0.95:
            break

    print('Rollout count =', rollout_count)

    action, _ = mcts.choose_action(temperature=0)
    return action


def profile_train():
    print('Running profiler on train()')
    file, memory = train()
    save_pkl(file, TEAM_NAME)
    with open('memory.pkl', 'wb') as f:
        pickle.dump(memory, f)



if __name__ == "__main__":
    #cProfile.run("profile_train()", "profile.prof")

    #file = train()
    #save_pkl(file, TEAM_NAME)
    my_pkl_file = load_pkl(TEAM_NAME)
    my_pkl_file= my_pkl_file.to(device)
    

   # my_mcts = AGZMCTS(random_state,c_puct=1.5,network=network,verbose=1)
    #print(my_mcts.choose_action(0.4))
    # Choose move functions when called in the game_mechanics expect only a state
    # argument, here is an example of how you can pass a pkl file and an initialized
    # mcts tree
    def choose_move_no_network(state: State) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_pkl_file, mcts=None)

    check_submission(
        TEAM_NAME, choose_move_no_network
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # Play a game against against your bot!
    play_go(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )