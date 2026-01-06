""" Plot training progress from training log """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_training_progress(log_file='training_log.csv', output_file='training_progress.png'):
    """ Plot training progress from CSV log file """

    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found!")
        print("Make sure training has started and created the log file.")
        return

    # Load the data
    try:
        df = pd.read_csv(log_file)
    except pd.errors.EmptyDataError:
        print(f"Error: Log file '{log_file}' is empty!")
        return

    if len(df) == 0:
        print("No data in log file yet. Start training first!")
        return

    print(f"Loaded {len(df)} episodes from {log_file}")
    print(f"Episode range: {df['episode'].min()} - {df['episode'].max()}")
    print(f"Current running mean reward: {df['running_mean_reward'].iloc[-1]:.2f}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Episode rewards (raw)
    ax1.plot(df['episode'], df['episode_reward'], alpha=0.3, color='blue', linewidth=0.5)

    # Add smoothed version (rolling average over 50 episodes)
    if len(df) >= 50:
        smoothed = df['episode_reward'].rolling(window=50, center=True).mean()
        ax1.plot(df['episode'], smoothed, color='blue', linewidth=2, label='50-episode moving average')

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if len(df) >= 50:
        ax1.legend()

    # Plot 2: Running mean reward
    ax2.plot(df['episode'], df['running_mean_reward'], color='green', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(df['episode'], df['running_mean_reward'], 0,
                      where=(df['running_mean_reward'] >= 0),
                      color='green', alpha=0.2, label='Winning')
    ax2.fill_between(df['episode'], df['running_mean_reward'], 0,
                      where=(df['running_mean_reward'] < 0),
                      color='red', alpha=0.2, label='Losing')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Running Mean Reward (0.99 decay)', fontsize=12)
    ax2.set_title('Running Mean Reward', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add text with statistics
    stats_text = f"Episodes: {len(df)}\n"
    stats_text += f"Latest reward: {df['episode_reward'].iloc[-1]:.1f}\n"
    stats_text += f"Running mean: {df['running_mean_reward'].iloc[-1]:.2f}\n"
    stats_text += f"Best episode: {df['episode_reward'].max():.1f}\n"
    stats_text += f"Worst episode: {df['episode_reward'].min():.1f}"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Allow specifying log file as command line argument
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'training_log.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'training_progress.png'

    plot_training_progress(log_file, output_file)
