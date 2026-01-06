""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym
import ale_py  # Required for ALE environments
from pong_utils import D, H, num_actions, softmax, prepro, discount_rewards, policy_forward, policy_backward

# hyperparameters
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# imitation learning parameters
use_imitation = False # whether to use human demonstrations
imitation_episodes = 1000 # how many episodes to train on human data before switching to RL
imitation_learning_rate = 1e-1 # learning rate for imitation learning (much higher!)
imitation_batch_size = 200 # larger batches for more stable gradients

# model initialization
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(num_actions, H) / np.sqrt(H) # 3 outputs for NOOP/UP/DOWN
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def load_human_demonstrations():
  """ Load human demonstration data from the most recent recording """
  import glob
  import os
  
  # Find the most recent human demonstration file
  demo_files = glob.glob("human_pong/human_pong_*.npz")
  if not demo_files:
    print("No human demonstration files found!")
    return None
    
  latest_file = max(demo_files, key=os.path.getctime)
  print(f"Loading human demonstrations from: {latest_file}")
  
  data = np.load(latest_file)
  frame_diffs = data['frame_diffs']
  actions = data['actions']
  
  print(f"Loaded {len(frame_diffs)} timesteps from human demonstration")
  print(f"Action distribution: {np.bincount(actions)}")
  
  return frame_diffs, actions

def imitation_learning_step(frame_diffs, actions):
  """ Perform one step of imitation learning using a more direct approach """
  # Use ALL actions: 0=NOOP, 2=UP, 3=DOWN (filter out FIRE actions 1,4,5)
  valid_mask = (actions == 0) | (actions == 2) | (actions == 3)
  if np.sum(valid_mask) == 0:
    print("Warning: No valid actions found in demonstration data!")
    return 0.0
    
  valid_frame_diffs = frame_diffs[valid_mask]
  valid_actions = actions[valid_mask]
  
  # Convert actions to one-hot: 0->[1,0,0], 2->[0,1,0], 3->[0,0,1]
  y_true = np.zeros((len(valid_actions), 3))
  y_true[valid_actions == 0, 0] = 1  # NOOP
  y_true[valid_actions == 2, 1] = 1  # UP  
  y_true[valid_actions == 3, 2] = 1  # DOWN
  
  # Use all data at once for stronger learning signal
  X = valid_frame_diffs.T  # (6400, num_samples)
  y = y_true.T  # (3, num_samples)
  
  # Forward pass
  h = np.dot(model['W1'], X)  # (200, num_samples)
  h[h < 0] = 0  # ReLU
  logits = np.dot(model['W2'], h)  # (3, num_samples)
  p = softmax(logits)  # (3, num_samples)
  
  # Compute loss (categorical cross-entropy)
  loss = -np.mean(y * np.log(p + 1e-8))
  
  # Compute gradients
  dlogits = p - y  # (3, num_samples)
  dW2 = np.dot(dlogits, h.T) / len(valid_actions)  # (3, 200)
  dh = np.dot(model['W2'].T, dlogits)  # (200, num_samples)
  dh[h <= 0] = 0  # backprop through ReLU
  dW1 = np.dot(dh, X.T) / len(valid_actions)  # (200, 6400)
  
  # Update model with much stronger learning
  model['W1'] -= imitation_learning_rate * dW1
  model['W2'] -= imitation_learning_rate * dW2
  
  return loss

env = gym.make("ALE/Pong-v5")
observation, info = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

# Setup logging
import csv
import os
log_file = 'training_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'episode_reward', 'running_mean_reward'])

# Load human demonstrations for imitation learnings
human_data = None
if use_imitation:
  human_data = load_human_demonstrations()
  if human_data is not None:
    frame_diffs, actions = human_data
    print("Starting imitation learning phase...")
    
    # Show data quality
    valid_actions = (actions == 0) | (actions == 2) | (actions == 3)
    noop_count = np.sum(actions == 0)
    up_count = np.sum(actions == 2)
    down_count = np.sum(actions == 3)
    print(f"Valid actions: {np.sum(valid_actions)}/{len(actions)} ({100*np.sum(valid_actions)/len(actions):.1f}%)")
    print(f"Action breakdown: NOOP={noop_count}, UP={up_count}, DOWN={down_count}")
    
    for i in range(imitation_episodes):
      loss = imitation_learning_step(frame_diffs, actions)
      
      # Show gradient norms and model statistics
      if (i + 1) % 10 == 0:
        w1_norm = np.linalg.norm(model['W1'])
        w2_norm = np.linalg.norm(model['W2'])
        w1_change = np.linalg.norm(model['W1'] - getattr(imitation_learning_step, 'prev_W1', model['W1']))
        w2_change = np.linalg.norm(model['W2'] - getattr(imitation_learning_step, 'prev_W2', model['W2']))
        print(f"Imitation episode {i+1}/{imitation_episodes}, loss: {loss:.4f}, W1_norm: {w1_norm:.3f}, W2_norm: {w2_norm:.3f}, W1_change: {w1_change:.6f}, W2_change: {w2_change:.6f}")
        imitation_learning_step.prev_W1 = model['W1'].copy()
        imitation_learning_step.prev_W2 = model['W2'].copy()
      else:
        print(f"Imitation episode {i+1}/{imitation_episodes}, loss: {loss:.4f}")
      
      # Test the policy after every 10 episodes
      if (i + 1) % 10 == 0:
        test_obs, _ = env.reset()
        test_prev_x = None
        test_reward = 0
        test_actions = []
        for _ in range(100):  # Test for 100 steps
          cur_x = prepro(test_obs)
          x = cur_x - test_prev_x if test_prev_x is not None else np.zeros(D)
          test_prev_x = cur_x
          aprob, _ = policy_forward(x, model)
          # Sample action from 3-class distribution: 0=NOOP, 2=UP, 3=DOWN
          action_probs = aprob
          action = np.random.choice([0, 2, 3], p=action_probs)
          test_actions.append(action)
          test_obs, reward, terminated, truncated, _ = env.step(action)
          test_reward += reward
          if terminated or truncated:
            break
        
        # Show action distribution in test
        action_counts = np.bincount(test_actions, minlength=6)
        print(f"  Test reward after episode {i+1}: {test_reward}, actions: NOOP={action_counts[0]}, UP={action_counts[2]}, DOWN={action_counts[3]}")
        
    print("Imitation learning complete! Switching to reinforcement learning...")
  else:
    print("No human data found, starting with random policy...")
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x, model)
  # Sample action from 3-class distribution: 0=NOOP, 2=UP, 3=DOWN
  action = np.random.choice([0, 2, 3], p=aprob)

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  # Create one-hot target for the action taken
  y = np.zeros(3)
  if action == 0: y[0] = 1  # NOOP
  elif action == 2: y[1] = 1  # UP
  elif action == 3: y[2] = 1  # DOWN
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

  # step the environment and get new measurements
  observation, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp, epx, model)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print(f'resetting env. episode reward total was {reward_sum:.3f}. running mean: {running_reward:.3f}')

    # Log to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode_number, reward_sum, running_reward])

    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation, info = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(f'ep {episode_number}: game finished, reward: {reward:.3f}' + ('' if reward == -1 else ' !!!!!!!!'))