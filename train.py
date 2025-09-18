""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym
import ale_py  # Required for ALE environments

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# imitation learning parameters
use_imitation = True # whether to use human demonstrations
imitation_episodes = 5 # how many episodes to train on human data before switching to RL
imitation_learning_rate = 1e-3 # learning rate for imitation learning (usually higher than RL)

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float64).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

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
  """ Perform one step of imitation learning using supervised learning """
  # Convert actions to binary: 2 (UP) -> 1, 3 (DOWN) -> 0
  y_true = (actions == 2).astype(np.float64)
  
  # Forward pass
  h = np.dot(model['W1'], frame_diffs.T)
  h[h < 0] = 0  # ReLU
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  
  # Compute loss (binary cross-entropy)
  loss = -np.mean(y_true * np.log(p + 1e-8) + (1 - y_true) * np.log(1 - p + 1e-8))
  
  # Compute gradients
  dlogp = p - y_true
  dW2 = np.dot(h, dlogp) / len(frame_diffs)
  dh = np.outer(dlogp, model['W2'])
  dh[h <= 0] = 0  # backprop through ReLU
  dW1 = np.dot(frame_diffs.T, dh) / len(frame_diffs)
  
  # Update model
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

# Load human demonstrations for imitation learning
human_data = None
if use_imitation:
  human_data = load_human_demonstrations()
  if human_data is not None:
    frame_diffs, actions = human_data
    print("Starting imitation learning phase...")
    for i in range(imitation_episodes):
      loss = imitation_learning_step(frame_diffs, actions)
      print(f"Imitation episode {i+1}/{imitation_episodes}, loss: {loss:.4f}")
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
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

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
    grad = policy_backward(eph, epdlogp)
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
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation, info = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(f'ep {episode_number}: game finished, reward: {reward:.3f}' + ('' if reward == -1 else ' !!!!!!!!'))