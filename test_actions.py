import time, numpy as np, gymnasium as gym
import ale_py  # Required for ALE environments
from gymnasium.utils.play import play

# Make Atari Pong
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# Keyboard → Atari actions
keys_to_action = {" ": 1, "ArrowUp": 2, "ArrowDown": 3}  # up arrow=UP, down arrow=DOWN
action_counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

def cb(obs_t, obs_tp1, act, rew, terminated, truncated, info):
    action_counts[str(act)] += 1
    print(f"Action: {act}, Reward: {rew}, Terminated: {terminated}")
    
    # Print action counts every 50 steps
    if sum(action_counts.values()) % 50 == 0:
        print(f"Action counts so far: {action_counts}")

print("Press UP arrow for UP, DOWN arrow for DOWN, space for FIRE")
print("Press Esc to exit")
print("=" * 50)

try:
    play(env, keys_to_action=keys_to_action, noop=0, fps=10, zoom=2.0, callback=cb)
finally:
    env.close()

print("\nFinal action counts:")
print(action_counts)
