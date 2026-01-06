"""
Watch trained agents play Pong with video recording capability.
Supports multiple model files and generates clearly named videos for each.
"""
import numpy as np
import pickle
import gymnasium as gym
import ale_py  # Required for ALE environments
from gymnasium.wrappers import RecordVideo
import time
import glob
import os

# Configuration - List of models to test
MODEL_PATHS = [
    "save-6plus.p",
    "save-13plus.p", 
    "save.p"  # Add any other models you want to test
]

# Filter to only existing models
EXISTING_MODELS = [path for path in MODEL_PATHS if os.path.exists(path)]

if not EXISTING_MODELS:
    print("No model files found! Please check that your .p files exist.")
    exit(1)

print(f"Found {len(EXISTING_MODELS)} model files to test:")
for model_path in EXISTING_MODELS:
    print(f"  - {model_path}")
print()

# Configuration
RECORD_VIDEO = True  # Set to False if you don't want video recording
EPISODES_TO_WATCH = 3  # Number of episodes to watch per model
RENDER_MODE = "human" if not RECORD_VIDEO else "rgb_array"
FPS = 30  # Frames per second for display

# Model parameters (should match training)
D = 80 * 80  # input dimensionality: 80x80 grid
H = 200      # hidden layer size
num_actions = 3  # NOOP, UP, DOWN

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def prepro(I):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float64).ravel()

def policy_forward(x, model):
    """Forward pass through the policy network"""
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logits = np.dot(model['W2'], h)  # (3,) - logits for NOOP/UP/DOWN
    p = softmax(logits)  # (3,) - probabilities for NOOP/UP/DOWN
    return p, h

def get_model_name(model_path):
    """Extract a clean model name from the file path for video naming"""
    # Remove .p extension and any path components
    name = os.path.basename(model_path).replace('.p', '')
    return name

def test_model(model_path):
    """Test a single model and return performance statistics"""
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_path}")
    print(f"{'='*60}")
    
    # Load the model
    try:
        model = pickle.load(open(model_path, 'rb'))
        print(f"Model loaded successfully!")
        print(f"Model keys: {list(model.keys())}")
        print(f"W1 shape: {model['W1'].shape}")
        print(f"W2 shape: {model['W2'].shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create environment
    print("Creating Pong environment...")
    env = gym.make("ALE/Pong-v5", render_mode=RENDER_MODE)
    
    # Wrap with video recording if requested
    model_name = get_model_name(model_path)
    if RECORD_VIDEO:
        env = RecordVideo(
            env, 
            video_folder="agent_videos", 
            name_prefix=f"agent-{model_name}", 
            episode_trigger=lambda ep: True  # Record every episode
        )
        print(f"Video recording enabled - videos will be saved with prefix 'agent-{model_name}'")
    
    # Watch the agent play
    print(f"\nWatching agent play {EPISODES_TO_WATCH} episodes...")
    print("Actions: 0=NOOP, 2=UP, 3=DOWN")
    print("-" * 50)
    
    total_reward = 0
    wins = 0
    games_played = 0
    all_episode_rewards = []
    
    for episode in range(EPISODES_TO_WATCH):
        observation, info = env.reset()
        prev_x = None
        episode_reward = 0
        step_count = 0
        episode_actions = []
        
        print(f"\nEpisode {episode + 1}/{EPISODES_TO_WATCH}")
        
        while True:
            # Preprocess the observation
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x
            
            # Get action probabilities from the policy
            aprob, h = policy_forward(x, model)
            
            # Choose action (deterministic - use most likely action)
            action_idx = np.argmax(aprob)  # 0, 1, or 2
            action = [0, 2, 3][action_idx]  # Convert to actual Atari actions
            
            episode_actions.append(action)
            
            # Take the action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            # Print game results
            if reward != 0:  # Pong gives +1 or -1 when a game ends
                games_played += 1
                if reward > 0:
                    wins += 1
                    print(f"  Game {games_played}: WON! (+{reward})")
                else:
                    print(f"  Game {games_played}: Lost ({reward})")
            
            # Add small delay for human viewing (only if not recording video)
            if not RECORD_VIDEO and RENDER_MODE == "human":
                time.sleep(1.0 / FPS)
            
            if done:
                break
        
        # Episode summary
        action_counts = np.bincount(episode_actions, minlength=6)
        total_reward += episode_reward
        all_episode_rewards.append(episode_reward)
        
        print(f"  Episode reward: {episode_reward}")
        print(f"  Steps taken: {step_count}")
        print(f"  Action distribution: NOOP={action_counts[0]}, UP={action_counts[2]}, DOWN={action_counts[3]}")
        print(f"  Final action probs: NOOP={aprob[0]:.3f}, UP={aprob[1]:.3f}, DOWN={aprob[2]:.3f}")
    
    # Close environment
    env.close()
    
    # Calculate statistics
    avg_reward = total_reward / EPISODES_TO_WATCH
    win_rate = (wins / games_played * 100) if games_played > 0 else 0
    
    # Model summary
    print(f"\n{'-'*50}")
    print(f"MODEL SUMMARY: {model_path}")
    print(f"{'-'*50}")
    print(f"Episodes played: {EPISODES_TO_WATCH}")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Episode rewards: {all_episode_rewards}")
    print(f"Games won: {wins}/{games_played} ({win_rate:.1f}%)")
    
    return {
        'model_path': model_path,
        'model_name': model_name,
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'wins': wins,
        'games_played': games_played,
        'episode_rewards': all_episode_rewards
    }

# Main execution - test all models
all_results = []
for model_path in EXISTING_MODELS:
    result = test_model(model_path)
    if result:
        all_results.append(result)

# Final comparison summary
print(f"\n{'='*80}")
print("FINAL COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<20} {'Avg Reward':<12} {'Win Rate':<10} {'Total Games':<12} {'Videos'}")
print(f"{'-'*80}")

for result in all_results:
    model_name = result['model_name']
    avg_reward = result['avg_reward']
    win_rate = result['win_rate']
    games_played = result['games_played']
    
    print(f"{model_name:<20} {avg_reward:<12.2f} {win_rate:<10.1f}% {games_played:<12} agent-{model_name}-episode-*.mp4")

if RECORD_VIDEO:
    print(f"\nAll videos saved to 'agent_videos' folder with clear naming:")
    for result in all_results:
        model_name = result['model_name']
        print(f"  - agent-{model_name}-episode-0.mp4, agent-{model_name}-episode-1.mp4, ...")

print("\nDone! You can now compare the performance of different models.")
