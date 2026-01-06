# Atari Pong RL Training

A reinforcement learning project that trains an AI agent to play Atari Pong using a combination of imitation learning and policy gradient methods.

## Overview

This project implements a two-phase training approach:

1. **Imitation Learning Phase**: The agent learns from human gameplay demonstrations using supervised learning
2. **Policy Gradient Phase**: The agent continues improving using the REINFORCE algorithm with advantage estimation

The implementation uses a simple 2-layer neural network built from scratch with NumPy (no PyTorch/TensorFlow), making it easy to understand the fundamentals of RL.

## Features

- Human demonstration recording with keyboard controls
- Imitation learning from human gameplay
- Policy gradient reinforcement learning (REINFORCE)
- Frame preprocessing (cropping, downsampling, binarization, differencing)
- Model checkpointing and resumable training
- Video recording of gameplay sessions
- **Synthetic demonstration generation from trained models**
- **Knowledge transfer experiments with varying data amounts**

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Record Human Demonstrations

Use the interactive play script to record human gameplay:

```bash
python play.py
```

**Controls:**
- Arrow keys or WASD to move paddle
- Spacebar for no action
- The script saves gameplay data to `human_pong/` directory

### 2. Train the Agent

Run the training script to train the agent:

```bash
python train.py
```

The training process:
- Loads human demonstrations from the most recent `.npz` file (if `use_imitation = True`)
- Runs 1000 episodes of imitation learning (if enabled)
- Continues with policy gradient RL indefinitely
- Saves checkpoints to `save.p` every 100 episodes
- Prints episode statistics (episode number, reward, running mean reward)
- Logs all episodes to `training_log.csv` for later analysis

Training will automatically resume from `save.p` if `resume = True`.

### Monitor Training Progress

While training is running (or after), visualize the progress:

```bash
python plot_training.py
```

This will:
- Load data from `training_log.csv`
- Generate plots showing:
  - Episode rewards over time (with 50-episode moving average)
  - Running mean reward (with winning/losing regions highlighted)
  - Summary statistics
- Save visualization to `training_progress.png`
- Display the plot interactively

You can run this anytime to check training progress without interrupting the training process.

### 3. Test Action Mappings (Optional)

Verify keyboard-to-action mappings:

```bash
python test_actions.py
```

## Knowledge Transfer Experiments

Once you have a trained model, you can study how efficiently knowledge transfers through imitation learning.

### 1. Generate Synthetic Demonstrations

Use your trained model to generate demonstration data:

```bash
python generate_demonstrations.py
```

This will:
- Load the trained model from `save.p`
- Play multiple episodes using the trained policy
- Save gameplay data to `model_demonstrations/` directory
- Print statistics about the generated demonstrations

You can edit the script to change `num_episodes` to generate more or fewer demonstrations.

### 2. Train New Models with Varying Data Amounts

Experiment with different amounts of demonstration data:

```bash
# Train with default fractions (10%, 25%, 50%, 75%, 100%)
python train_from_model_demos.py

# Custom data fractions
python train_from_model_demos.py --fractions 0.05 0.1 0.2 0.5 1.0

# Include some RL episodes after imitation
python train_from_model_demos.py --rl-episodes 50

# Change number of imitation episodes
python train_from_model_demos.py --imitation-episodes 500
```

This will:
- Train separate models with different amounts of demonstration data
- Evaluate each model periodically during training
- Save models and results to `experiments/` directory
- Print training progress and final statistics

### 3. Analyze Results

Compare the learning efficiency across different data amounts:

```bash
python analyze_results.py
```

This will:
- Load all experiment results from `experiments/` directory
- Print summary statistics for each experiment
- Generate comparison plots showing:
  - Training loss over time
  - Action prediction accuracy
  - Evaluation rewards
  - Final performance vs. data amount
- Save plots to `experiments/learning_curves_comparison.png`

### Experiment Goals

This setup lets you answer questions like:
- **Data efficiency**: How much demonstration data is needed to reach good performance?
- **Diminishing returns**: Does using 100% vs 50% of data make a significant difference?
- **Learning speed**: Do models with more data converge faster?
- **Performance ceiling**: Is there a minimum data amount below which models fail to learn?

## Architecture

### Neural Network

```
Input: 6400 (80×80 flattened frame difference)
   ↓
Hidden Layer: 200 neurons + ReLU
   ↓
Output: 1 neuron + Sigmoid
   ↓
Action: Binary decision (UP/RIGHT vs DOWN/LEFT)
```

### Frame Preprocessing Pipeline

1. **Crop**: Remove UI elements (160×160 region)
2. **Downsample**: 2×2 downsampling (80×80)
3. **Binarize**: Convert to binary (paddles/ball = 1, background = 0)
4. **Difference**: Compute frame difference to capture motion

### Training Algorithm

**Imitation Learning:**
- Loss: Binary cross-entropy
- Learning rate: 0.1
- Episodes: 1000

**Policy Gradient (REINFORCE):**
- Algorithm: REINFORCE with advantage estimation
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Batch size: 10 episodes
- Optimizer: RMSProp (decay=0.99)

## File Structure

```
.
├── train.py                      # Main training script
├── play.py                       # Interactive human gameplay recorder
├── test_actions.py               # Action mapping testing utility
├── generate_demonstrations.py    # Generate synthetic demos from trained model
├── train_from_model_demos.py    # Train new models with varying data amounts
├── analyze_results.py            # Analyze and visualize experiment results
├── plot_training.py              # Plot training progress from logs
├── requirements.txt              # Python dependencies
├── save.p                        # Model checkpoint (created during training)
├── training_log.csv              # Training logs (created during training)
├── training_progress.png         # Training visualization (created by plot_training.py)
├── human_pong/                   # Human demonstration data (created by play.py)
│   ├── *.npz                     # Gameplay data files
│   └── *.mp4                     # Gameplay videos
├── model_demonstrations/         # Synthetic demos (created by generate_demonstrations.py)
│   └── *.npz                     # Model-generated gameplay data
└── experiments/                  # Experiment results (created by train_from_model_demos.py)
    ├── *_after_imitation.p       # Trained model checkpoints
    ├── *_results.npz             # Evaluation metrics
    └── learning_curves_comparison.png  # Results visualization
```

## Hyperparameters

Key hyperparameters in `train.py`:

```python
H = 200                    # Hidden layer neurons
batch_size = 10           # Episodes per batch
learning_rate = 1e-4      # RL learning rate
imitation_lr = 1e-1       # Imitation learning rate
gamma = 0.99              # Discount factor
decay_rate = 0.99         # RMSProp decay
imitation_episodes = 1000 # Imitation training episodes
```

## How It Works

### Phase 1: Imitation Learning

The agent learns from human demonstrations:
1. Loads recorded human gameplay from `.npz` files
2. Trains network to predict human actions using supervised learning
3. Uses binary cross-entropy loss
4. Runs for 1000 episodes to establish baseline policy

### Phase 2: Policy Gradient RL

The agent improves through self-play:
1. Collects episodes by playing games
2. Computes discounted rewards and advantages
3. Updates policy to increase probability of good actions
4. Uses batch updates every 10 episodes
5. Continues indefinitely to maximize performance

### Why Frame Differencing?

Computing the difference between consecutive frames captures motion information, which is crucial for understanding:
- Ball velocity and direction
- Paddle movement
- Game dynamics

This is more informative than raw frames for decision-making.

## Expected Results

- **Imitation Phase**: Agent learns basic Pong gameplay from human demonstrations
- **RL Phase**: Agent improves beyond human performance through self-play
- Training progress is shown as running mean reward (target: positive scores)

## Dependencies

- `gymnasium==1.2.0` - RL environment framework
- `ale-py==0.11.2` - Atari Learning Environment
- `numpy==2.2.6` - Numerical computation
- `opencv-python==4.12.0.88` - Computer vision
- `moviepy==2.2.1` - Video processing
- `pygame==2.6.1` - Game rendering
- See [requirements.txt](requirements.txt) for complete list

## License

This is an educational project for learning reinforcement learning concepts.

## Acknowledgments

Based on Andrej Karpathy's "Deep Reinforcement Learning: Pong from Pixels" blog post and extended with imitation learning capabilities.
