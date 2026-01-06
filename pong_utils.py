"""
Shared utility functions for Pong RL training.
Contains common preprocessing, activation functions, and helpers.
"""
import numpy as np

# Constants
D = 80 * 80  # input dimensionality: 80x80 grid
H = 200  # number of hidden layer neurons
num_actions = 3  # NOOP, UP, DOWN

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def prepro(I):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float64).ravel()

def discount_rewards(r, gamma=0.99):
    """Take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x, model):
    """Forward pass through the policy network

    Args:
        x: Input state (6400,)
        model: Dict with 'W1' (200, 6400) and 'W2' (3, 200)

    Returns:
        p: Action probabilities (3,) for NOOP/UP/DOWN
        h: Hidden layer activations (200,)
    """
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logits = np.dot(model['W2'], h)  # (3,) - logits for NOOP/UP/DOWN
    p = softmax(logits)  # (3,) - probabilities for NOOP/UP/DOWN
    return p, h

def policy_backward(eph, epdlogp, epx, model):
    """Backward pass through the policy network

    Args:
        eph: Array of hidden states (timesteps, 200)
        epdlogp: Array of policy gradients (timesteps, 3)
        epx: Array of input states (timesteps, 6400)
        model: Dict with 'W1' and 'W2'

    Returns:
        dict with gradient for 'W1' and 'W2'
    """
    dW2 = np.dot(eph.T, epdlogp).T  # (200, timesteps) @ (timesteps, 3) = (200, 3) -> (3, 200)
    dh = np.dot(epdlogp, model['W2'])  # (timesteps, 3) @ (3, 200) = (timesteps, 200)
    dh[eph <= 0] = 0  # backprop through ReLU
    dW1 = np.dot(dh.T, epx)  # (200, timesteps) @ (timesteps, 6400) = (200, 6400)
    return {'W1': dW1, 'W2': dW2}
