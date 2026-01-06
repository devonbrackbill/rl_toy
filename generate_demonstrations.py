""" Generate demonstration recordings from a trained model """
import numpy as np
import pickle
import gymnasium as gym
import ale_py
from datetime import datetime
import os

# hyperparameters
H = 200  # number of hidden layer neurons
D = 80 * 80  # input dimensionality: 80x80 grid
num_episodes = 100  # number of episodes to record
output_dir = "model_demonstrations"  # directory to save demonstrations

# Load the trained model
print("Loading trained model from save 18plus.p...")
model = pickle.load(open('save 18plus.p', 'rb'))
print(f"Model loaded. W1 shape: {model['W1'].shape}, W2 shape: {model['W2'].shape}")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2,::2,0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float64).ravel()

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0  # ReLU nonlinearity
    logits = np.dot(model['W2'], h)  # (3,) - logits for NOOP/UP/DOWN
    p = softmax(logits)  # (3,) - probabilities for NOOP/UP/DOWN
    return p, h

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize environment
env = gym.make("ALE/Pong-v5")

# Storage for all demonstrations
all_frame_diffs = []
all_actions = []
all_rewards = []
all_dones = []
episode_rewards = []

print(f"\nGenerating {num_episodes} demonstration episodes...")
print("-" * 60)

for episode in range(num_episodes):
    observation, info = env.reset()
    prev_x = None
    episode_reward = 0
    episode_steps = 0

    # Storage for this episode
    episode_frame_diffs = []
    episode_actions = []
    episode_rewards_list = []
    episode_dones = []

    while True:
        # Preprocess the observation
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # Get action from policy
        aprob, h = policy_forward(x)
        # Choose action (deterministic - use most likely action)
        action_idx = np.argmax(aprob)  # 0, 1, or 2
        action = [0, 2, 3][action_idx]  # Convert to actual Atari actions (NOOP, UP, DOWN)

        # Store the data
        episode_frame_diffs.append(x)
        episode_actions.append(action)

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_rewards_list.append(reward)
        episode_dones.append(done)
        episode_reward += reward
        episode_steps += 1

        if done:
            break

    # Add episode data to overall storage
    all_frame_diffs.extend(episode_frame_diffs)
    all_actions.extend(episode_actions)
    all_rewards.extend(episode_rewards_list)
    all_dones.extend(episode_dones)
    episode_rewards.append(episode_reward)

    action_counts = np.bincount(episode_actions, minlength=6)
    print(f"Episode {episode+1}/{num_episodes}: "
          f"Steps={episode_steps}, Reward={episode_reward:.0f}, "
          f"Action distribution: NOOP={action_counts[0]}, UP={action_counts[2]}, DOWN={action_counts[3]}")

env.close()

# Convert to numpy arrays
all_frame_diffs = np.array(all_frame_diffs)
all_actions = np.array(all_actions)
all_rewards = np.array(all_rewards)
all_dones = np.array(all_dones)

# Print statistics
print("-" * 60)
print(f"\nGeneration complete!")
print(f"Total timesteps: {len(all_frame_diffs)}")
print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
print(f"Average episode length: {len(all_frame_diffs) / num_episodes:.1f} steps")
print(f"Action distribution: NOOP (action 0)={np.sum(all_actions==0)}, "
      f"UP (action 2)={np.sum(all_actions==2)}, "
      f"DOWN (action 3)={np.sum(all_actions==3)}")
print(f"Win rate: {np.sum(np.array(episode_rewards) > 0) / len(episode_rewards) * 100:.1f}%")

# Save the demonstrations
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{output_dir}/model_demo_{num_episodes}ep_{timestamp}.npz"
np.savez_compressed(
    filename,
    frame_diffs=all_frame_diffs,
    actions=all_actions,
    rewards=all_rewards,
    dones=all_dones,
    episode_rewards=episode_rewards
)

print(f"\nDemonstrations saved to: {filename}")
print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
