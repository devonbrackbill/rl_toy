""" Train a new model using varying amounts of model-generated demonstrations """
import numpy as np
import pickle
import gymnasium as gym
import ale_py
import glob
import os
import argparse
from pong_utils import D, H, num_actions, softmax, prepro, discount_rewards, policy_forward, policy_backward

# hyperparameters
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
render = False

# imitation learning parameters
imitation_episodes = 1000  # how many episodes to train on model data before switching to RL
imitation_learning_rate = 1e-1  # learning rate for imitation learning
eval_frequency = 10  # evaluate every N imitation episodes

def load_model_demonstrations(data_fraction=1.0, held_out_fraction=0.2):
    """ Load model-generated demonstration data with fixed held-out set

    Args:
        data_fraction: Fraction of TRAINING data to use (0.0 to 1.0)
        held_out_fraction: Fraction of total data to hold out for validation (default: 0.2)
                          This held-out set is FIXED across all runs
    """
    demo_files = glob.glob("model_demonstrations/model_demo_*.npz")
    if not demo_files:
        print("No model demonstration files found!")
        return None

    latest_file = max(demo_files, key=os.path.getctime)
    print(f"Loading model demonstrations from: {latest_file}")

    data = np.load(latest_file)
    all_frame_diffs = data['frame_diffs']
    all_actions = data['actions']

    # First, split into train and held-out sets (fixed split)
    num_total = len(all_frame_diffs)
    num_held_out = int(num_total * held_out_fraction)
    num_available_train = num_total - num_held_out

    # Use fixed random seed for reproducible train/held-out split
    rng = np.random.RandomState(42)
    indices = rng.permutation(num_total)
    train_pool_indices = indices[:num_available_train]
    held_out_indices = indices[num_available_train:]

    held_out_frame_diffs = all_frame_diffs[held_out_indices]
    held_out_actions = all_actions[held_out_indices]

    # Now sample from the training pool according to data_fraction
    if data_fraction < 1.0:
        num_train_samples = int(num_available_train * data_fraction)
        sampled_indices = rng.choice(train_pool_indices, num_train_samples, replace=False)
        train_frame_diffs = all_frame_diffs[sampled_indices]
        train_actions = all_actions[sampled_indices]
        print(f"Using {data_fraction*100:.1f}% of training data: {num_train_samples}/{num_available_train} samples")
    else:
        train_frame_diffs = all_frame_diffs[train_pool_indices]
        train_actions = all_actions[train_pool_indices]
        print(f"Using all training data: {num_available_train} samples")

    print(f"Fixed held-out set: {num_held_out} samples")
    print(f"Train action distribution: NOOP={np.sum(train_actions==0)}, UP={np.sum(train_actions==2)}, DOWN={np.sum(train_actions==3)}")
    print(f"Held-out action distribution: NOOP={np.sum(held_out_actions==0)}, UP={np.sum(held_out_actions==2)}, DOWN={np.sum(held_out_actions==3)}")

    return (train_frame_diffs, train_actions), (held_out_frame_diffs, held_out_actions)

def compute_accuracy(frame_diffs, actions, model):
    """ Compute accuracy on a dataset without updating the model """
    # Convert actions to one-hot encoding
    action_to_idx = {0: 0, 2: 1, 3: 2}
    y_indices = np.array([action_to_idx[a] for a in actions])
    num_samples = len(actions)

    # Create one-hot encoding (num_samples, 3)
    y_true = np.zeros((num_samples, 3))
    y_true[np.arange(num_samples), y_indices] = 1

    # Use all data at once
    X = frame_diffs.T  # (6400, num_samples)

    # Forward pass
    h = np.dot(model['W1'], X)  # (200, num_samples)
    h[h < 0] = 0  # ReLU
    logits = np.dot(model['W2'], h)  # (3, num_samples)
    p = softmax(logits)  # (3, num_samples)

    # Compute loss (categorical cross-entropy)
    # Clip probabilities to avoid log(0)
    p_clipped = np.clip(p, 1e-8, 1.0)
    loss = -np.mean(np.sum(y_true.T * np.log(p_clipped), axis=0))

    # Compute accuracy
    predictions = np.argmax(p, axis=0)  # (num_samples,)
    accuracy = np.mean(predictions == y_indices)

    return loss, accuracy

def imitation_learning_step(frame_diffs, actions, model):
    """ Perform one step of imitation learning """
    # Compute loss and accuracy
    loss, accuracy = compute_accuracy(frame_diffs, actions, model)

    # Convert actions to one-hot encoding
    action_to_idx = {0: 0, 2: 1, 3: 2}
    y_indices = np.array([action_to_idx[a] for a in actions])
    num_samples = len(actions)
    y_true = np.zeros((num_samples, 3))
    y_true[np.arange(num_samples), y_indices] = 1

    # Use all data at once
    X = frame_diffs.T  # (6400, num_samples)

    # Forward pass
    h = np.dot(model['W1'], X)  # (200, num_samples)
    h[h < 0] = 0  # ReLU
    logits = np.dot(model['W2'], h)  # (3, num_samples)
    p = softmax(logits)  # (3, num_samples)

    # Compute gradients
    dlogits = p - y_true.T  # (3, num_samples)
    dW2 = np.dot(dlogits, h.T) / num_samples  # (3, 200)
    dh = np.dot(model['W2'].T, dlogits)  # (200, num_samples)
    dh[h <= 0] = 0  # backprop through ReLU
    dW1 = np.dot(dh, X.T) / num_samples  # (200, 6400)

    # Update model
    model['W1'] -= imitation_learning_rate * dW1
    model['W2'] -= imitation_learning_rate * dW2

    return loss, accuracy

def evaluate_policy(model, env, num_episodes=20):
    """ Evaluate the current policy """
    total_reward = 0
    total_steps = 0

    for _ in range(num_episodes):
        observation, _ = env.reset()
        prev_x = None
        episode_reward = 0

        for step in range(10000):  # max steps per episode
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x

            aprob, _ = policy_forward(x, model)
            # Choose action (deterministic - use most likely action)
            action_idx = np.argmax(aprob)  # 0, 1, or 2
            action = [0, 2, 3][action_idx]  # Convert to actual Atari actions

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            total_steps += 1

            if terminated or truncated:
                break

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    return avg_reward, avg_steps

def train_with_data_fraction(data_fraction, experiment_name, num_rl_episodes=100):
    """ Train a model with a specific fraction of demonstration data

    Args:
        data_fraction: Fraction of demonstration data to use
        experiment_name: Name for saving checkpoints
        num_rl_episodes: Number of RL episodes to run after imitation learning
    """
    print("\n" + "="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Data fraction: {data_fraction*100:.1f}%")
    print("="*70)

    # Initialize fresh model
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(num_actions, H) / np.sqrt(H)  # 3 actions: NOOP, UP, DOWN

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

    # Load demonstration data with fixed held-out set
    demo_data = load_model_demonstrations(data_fraction)
    if demo_data is None:
        print("Failed to load demonstration data!")
        return

    (train_frame_diffs, train_actions), (held_out_frame_diffs, held_out_actions) = demo_data

    # Initialize environment
    env = gym.make("ALE/Pong-v5")

    # Evaluate initial random policy
    print("\nEvaluating initial random policy...")
    init_reward, init_steps = evaluate_policy(model, env, num_episodes=20)
    print(f"Initial policy: avg_reward={init_reward:.2f}, avg_steps={init_steps:.1f}")

    # Compute initial held-out accuracy
    init_held_out_loss, init_held_out_acc = compute_accuracy(held_out_frame_diffs, held_out_actions, model)
    print(f"Initial held-out: loss={init_held_out_loss:.4f}, accuracy={init_held_out_acc:.3f}")

    # Imitation learning phase
    print(f"\nStarting imitation learning for {imitation_episodes} episodes...")
    print("-" * 70)

    eval_results = []
    for i in range(imitation_episodes):
        train_loss, train_accuracy = imitation_learning_step(train_frame_diffs, train_actions, model)

        if (i + 1) % eval_frequency == 0:
            # Compute held-out accuracy
            held_out_loss, held_out_accuracy = compute_accuracy(held_out_frame_diffs, held_out_actions, model)

            # Evaluate current policy
            avg_reward, avg_steps = evaluate_policy(model, env, num_episodes=20)
            eval_results.append({
                'episode': i + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'held_out_loss': held_out_loss,
                'held_out_accuracy': held_out_accuracy,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps
            })
            print(f"Episode {i+1}/{imitation_episodes} | "
                  f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.3f} | "
                  f"Held-out - Loss: {held_out_loss:.4f}, Acc: {held_out_accuracy:.3f} | "
                  f"Eval reward: {avg_reward:.2f}")
        else:
            print(f"Episode {i+1}/{imitation_episodes} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.3f}")

    # Save model after imitation learning
    os.makedirs("experiments", exist_ok=True)
    checkpoint_path = f"experiments/{experiment_name}_after_imitation.p"
    pickle.dump(model, open(checkpoint_path, 'wb'))
    print(f"\nModel saved to: {checkpoint_path}")

    # Save evaluation results
    results_path = f"experiments/{experiment_name}_results.npz"
    np.savez(results_path,
             eval_results=np.array(eval_results),
             data_fraction=data_fraction,
             imitation_episodes=imitation_episodes)
    print(f"Results saved to: {results_path}")

    # Optional: Run some RL episodes
    if num_rl_episodes > 0:
        print(f"\nRunning {num_rl_episodes} RL episodes to see improvement...")
        observation, info = env.reset()
        prev_x = None
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        while episode_number < num_rl_episodes:
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x

            aprob, h = policy_forward(x, model)  # aprob is (3,)
            # Sample action from probability distribution
            action_idx = np.random.choice(3, p=aprob)
            action = [0, 2, 3][action_idx]

            xs.append(x)
            hs.append(h)
            # Compute gradient: one-hot encoding minus predicted probabilities
            y = np.zeros(3)
            y[action_idx] = 1.0
            dlogps.append(y - aprob)  # (3,)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            drs.append(reward)

            if done:
                episode_number += 1

                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)  # (timesteps, 3)
                epr = np.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []

                discounted_epr = discount_rewards(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= (np.std(discounted_epr) + 1e-8)

                epdlogp *= discounted_epr  # (timesteps, 3) * (timesteps, 1)
                grad = policy_backward(eph, epdlogp, epx, model)
                for k in model:
                    grad_buffer[k] += grad[k]

                if episode_number % batch_size == 0:
                    for k, v in model.items():
                        g = grad_buffer[k]
                        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                        grad_buffer[k] = np.zeros_like(v)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(f'RL episode {episode_number}/{num_rl_episodes}: reward={reward_sum:.0f}, running_mean={running_reward:.2f}')
                reward_sum = 0
                observation, info = env.reset()
                prev_x = None

    env.close()
    print(f"\nExperiment {experiment_name} complete!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train from model demonstrations with varying data amounts')
    parser.add_argument('--fractions', nargs='+', type=float,
                       default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                       help='Data fractions to experiment with (e.g., 0.01 0.05 0.1 0.25 0.5 1.0)')
    parser.add_argument('--rl-episodes', type=int, default=0,
                       help='Number of RL episodes to run after imitation (default: 0)')
    parser.add_argument('--imitation-episodes', type=int, default=500,
                       help='Number of imitation learning episodes (default: 500)')

    args = parser.parse_args()

    # Update global imitation_episodes if specified
    imitation_episodes = args.imitation_episodes

    # Run experiments with different data fractions
    for fraction in args.fractions:
        experiment_name = f"data_{int(fraction*100)}pct"
        train_with_data_fraction(fraction, experiment_name, args.rl_episodes)

    print("\n" + "="*70)
    print("All experiments complete!")
    print("Check the 'experiments/' directory for saved models and results.")
    print("="*70)
