import time, numpy as np, gymnasium as gym
import ale_py  # Required for ALE environments
from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play

# Make Atari Pong with a video-capable render mode
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
# Wrap to save MP4s of your play (every episode)
env = RecordVideo(env, video_folder="human_pong", name_prefix="human", episode_trigger=lambda ep: True)

# Keyboard → Atari actions (0=NOOP, 1=FIRE, 2=UP, 3=DOWN for training compatibility)
keys_to_action = {" ": 1, "w": 2, "s": 3}  # Changed "s": 5 to "s": 3
traj = {"frame_diffs": [], "actions": [], "rewards": [], "terminated": [], "truncated": []}

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float64).ravel()

def cb(obs_t, obs_tp1, act, rew, terminated, truncated, info):
    # Preprocess the current frame
    cur_x = prepro(obs_tp1)
    
    # Calculate frame difference (same as training script)
    if not hasattr(cb, 'prev_x'):
        cb.prev_x = None
    x = cur_x - cb.prev_x if cb.prev_x is not None else np.zeros(6400)
    cb.prev_x = cur_x
    
    # Store preprocessed frame difference and action
    traj["frame_diffs"].append(x)
    traj["actions"].append(act)
    traj["rewards"].append(rew)
    traj["terminated"].append(terminated)
    traj["truncated"].append(truncated)
    
    # Reset prev_x on episode end
    if terminated or truncated:
        cb.prev_x = None

# Play!  Esc closes the window.
try:
    play(env, keys_to_action=keys_to_action, noop=0, fps=10, zoom=2.0, callback=cb)
finally:
    # Properly close the environment to save the video
    env.close()

# Save your demonstration
print(f"Recording complete! Saved {len(traj['frame_diffs'])} timesteps")
print(f"Actions taken: {np.bincount(traj['actions'])}")
np.savez_compressed(f"human_pong/human_pong_{int(time.time())}.npz",
                    **{k: np.array(v) for k, v in traj.items()})
