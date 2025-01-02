# AlphaZero Go: 9x9 Board Implementation

This repository contains a custom implementation of AlphaZero for playing Go on a 9x9 board. It was developed as part of a competitive programming challenge with specific rules and constraints. The project demonstrates the integration of reinforcement learning and Monte Carlo Tree Search (MCTS) to build a powerful Go-playing agent.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Rules & Scoring](#rules--scoring)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

Go is a strategic board game where players aim to capture the most territory. While professional Go is played on a 19x19 board, this project focuses on a 9x9 board for computational efficiency. The implemented agent uses a combination of reinforcement learning and planning algorithms to make decisions.

### Key Features
- **Customizable Training Pipeline**: Train the agent using reinforcement learning, guided by the AlphaZero framework.
- **Efficient MCTS**: A Monte Carlo Tree Search implementation optimized for quick decision-making within a 1-second time limit.
- **Legal Move Enforcement**: Ensures all moves adhere to Go rules, including simple Ko.
- **Area Scoring with Komi**: Implements area scoring with a Komi of 7.5 to balance the advantage of playing first.
- **Modular Design**: Clean separation of logic for training, move selection, and game environment.

---

## Rules & Scoring

- **Board Size**: 9x9 grid.
- **Game Start**: Black always plays first.
- **Scoring**: Area scoring is used. Player score = number of stones on the board + empty territories they control. A Komi of 7.5 is subtracted from Black's final score.
- **Game End**: The game concludes when both players pass consecutively.
- **Move Selection Time**: The agent has a strict 1-second limit to select its move.

---

## Implementation Details

### Core Components

1. **Reinforcement Learning**:
   - Neural networks predict move probabilities and board value.
   - Training is done using self-play games to iteratively improve performance.

2. **Monte Carlo Tree Search (MCTS)**:
   - Guides the agent by simulating possible game outcomes.
   - Persistent tree structure across moves for efficient pruning and exploration.

3. **Game Environment**:
   - Built on a custom Go environment (`GoEnv`) with helper functions like `int_to_coord()` and `is_terminal()`.
   - Enforces game rules and provides a list of legal moves.

### File Structure

```plaintext
main/
│   main.py               # Entry point for training and gameplay
│   mcts.py               # MCTS implementation
│   train.py              # Training logic for reinforcement learning
│   utils.py              # Helper functions for Go rules and scoring
│   model.py              # Neural network architecture
│   config.py             # Configuration and hyperparameters
```

---

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/alphazero-go.git
   cd alphazero-go