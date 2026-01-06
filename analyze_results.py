""" Analyze and compare results from different data fraction experiments """
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results():
    """ Load all experiment results """
    results_files = glob.glob("experiments/*_results.npz")

    if not results_files:
        print("No experiment results found in experiments/ directory!")
        return None

    experiments = []
    for filepath in results_files:
        data = np.load(filepath, allow_pickle=True)
        experiment_name = os.path.basename(filepath).replace('_results.npz', '')

        experiments.append({
            'name': experiment_name,
            'data_fraction': float(data['data_fraction']),
            'imitation_episodes': int(data['imitation_episodes']),
            'eval_results': data['eval_results']
        })

    # Sort by data fraction
    experiments.sort(key=lambda x: x['data_fraction'])

    return experiments

def print_summary(experiments):
    """ Print summary statistics """
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"{'Data %':<10} {'Episodes':<12} {'Train Loss':<12} {'Train Acc':<12} {'Held-out Acc':<15} {'Final Reward':<15} {'Best Reward':<15}")
    print("-"*100)

    for exp in experiments:
        eval_results = exp['eval_results']
        if len(eval_results) > 0:
            # Get final and best metrics
            final = eval_results[-1]
            best_reward = max([r['avg_reward'] for r in eval_results])

            # Handle both old and new result formats
            train_loss = final.get('train_loss', final.get('loss', 0))
            train_acc = final.get('train_accuracy', final.get('accuracy', 0))
            held_out_acc = final.get('held_out_accuracy', 0)

            print(f"{exp['data_fraction']*100:<10.1f} "
                  f"{exp['imitation_episodes']:<12} "
                  f"{train_loss:<12.4f} "
                  f"{train_acc:<12.3f} "
                  f"{held_out_acc:<15.3f} "
                  f"{final['avg_reward']:<15.2f} "
                  f"{best_reward:<15.2f}")

    print("="*100)

def plot_learning_curves(experiments):
    """ Plot learning curves comparing different data fractions """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Loss over episodes
    ax = axes[0, 0]
    for exp in experiments:
        eval_results = exp['eval_results']
        episodes = [r['episode'] for r in eval_results]
        # Handle both old and new result formats
        losses = [r.get('train_loss', r.get('loss', 0)) for r in eval_results]
        ax.plot(episodes, losses, marker='o', label=f"{exp['data_fraction']*100:.0f}%", linewidth=2)
    ax.set_xlabel('Imitation Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss vs Episodes', fontsize=14, fontweight='bold')
    ax.legend(title='Data Used', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Training accuracy over episodes
    ax = axes[0, 1]
    for exp in experiments:
        eval_results = exp['eval_results']
        episodes = [r['episode'] for r in eval_results]
        # Handle both old and new result formats
        accuracies = [r.get('train_accuracy', r.get('accuracy', 0)) for r in eval_results]
        ax.plot(episodes, accuracies, marker='o', label=f"{exp['data_fraction']*100:.0f}%", linewidth=2)
    ax.set_xlabel('Imitation Episode', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy vs Episodes', fontsize=14, fontweight='bold')
    ax.legend(title='Data Used', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Held-out accuracy over episodes (NEW)
    ax = axes[0, 2]
    for exp in experiments:
        eval_results = exp['eval_results']
        episodes = [r['episode'] for r in eval_results]
        # Only plot if held-out accuracy is available
        held_out_accs = [r.get('held_out_accuracy', None) for r in eval_results]
        if any(acc is not None for acc in held_out_accs):
            # Replace None with 0 for plotting
            held_out_accs = [acc if acc is not None else 0 for acc in held_out_accs]
            ax.plot(episodes, held_out_accs, marker='o', label=f"{exp['data_fraction']*100:.0f}%", linewidth=2)
    ax.set_xlabel('Imitation Episode', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Held-out Accuracy vs Episodes', fontsize=14, fontweight='bold')
    ax.legend(title='Data Used', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Reward over episodes
    ax = axes[1, 0]
    for exp in experiments:
        eval_results = exp['eval_results']
        episodes = [r['episode'] for r in eval_results]
        rewards = [r['avg_reward'] for r in eval_results]
        ax.plot(episodes, rewards, marker='o', label=f"{exp['data_fraction']*100:.0f}%", linewidth=2)
    ax.set_xlabel('Imitation Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Evaluation Reward vs Episodes', fontsize=14, fontweight='bold')
    ax.legend(title='Data Used', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 5: Final performance vs data fraction
    ax = axes[1, 1]
    data_fractions = [exp['data_fraction'] * 100 for exp in experiments]
    final_rewards = [exp['eval_results'][-1]['avg_reward'] for exp in experiments]
    best_rewards = [max([r['avg_reward'] for r in exp['eval_results']]) for exp in experiments]

    x_pos = np.arange(len(data_fractions))
    width = 0.35
    ax.bar(x_pos - width/2, final_rewards, width, label='Final Reward', alpha=0.8)
    ax.bar(x_pos + width/2, best_rewards, width, label='Best Reward', alpha=0.8)
    ax.set_xlabel('Data Used (%)', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Final Performance vs Data Amount', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{f:.0f}" for f in data_fractions])
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Train vs Held-out accuracy comparison (NEW)
    ax = axes[1, 2]
    for exp in experiments:
        eval_results = exp['eval_results']
        if any('held_out_accuracy' in r for r in eval_results):
            final = eval_results[-1]
            train_acc = final.get('train_accuracy', final.get('accuracy', 0))
            held_out_acc = final.get('held_out_accuracy', 0)
            data_pct = exp['data_fraction'] * 100

            # We'll accumulate data and plot after the loop
            if not hasattr(ax, '_train_accs'):
                ax._train_accs = []
                ax._held_out_accs = []
                ax._data_pcts = []

            ax._train_accs.append(train_acc)
            ax._held_out_accs.append(held_out_acc)
            ax._data_pcts.append(data_pct)

    # Plot the accumulated data
    if hasattr(ax, '_train_accs'):
        x_pos = np.arange(len(ax._data_pcts))
        width = 0.35
        ax.bar(x_pos - width/2, ax._train_accs, width, label='Train Acc', alpha=0.8)
        ax.bar(x_pos + width/2, ax._held_out_accs, width, label='Held-out Acc', alpha=0.8)
        ax.set_xlabel('Data Used (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Final Train vs Held-out Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{f:.0f}" for f in ax._data_pcts])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No held-out data available', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Save figure
    output_path = "experiments/learning_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def compute_data_efficiency_metrics(experiments):
    """ Compute metrics about data efficiency """
    print("\n" + "="*80)
    print("DATA EFFICIENCY ANALYSIS")
    print("="*80)

    if len(experiments) < 2:
        print("Need at least 2 experiments to analyze data efficiency")
        return

    # Find minimum data needed to reach certain performance thresholds
    thresholds = [0, 5, 10, 15]

    for threshold in thresholds:
        print(f"\nTo reach avg reward >= {threshold}:")
        found_any = False
        for exp in experiments:
            eval_results = exp['eval_results']
            for result in eval_results:
                if result['avg_reward'] >= threshold:
                    found_any = True
                    print(f"  {exp['data_fraction']*100:>5.1f}% data: reached at episode {result['episode']}")
                    break
        if not found_any:
            print(f"  No experiments reached this threshold")

    print("\n" + "="*80)

if __name__ == "__main__":
    print("Loading experiment results...")
    experiments = load_experiment_results()

    if experiments is None or len(experiments) == 0:
        print("\nNo experiments found. Run train_from_model_demos.py first!")
        exit(1)

    print(f"Found {len(experiments)} experiments")

    # Print summary
    print_summary(experiments)

    # Compute efficiency metrics
    compute_data_efficiency_metrics(experiments)

    # Plot learning curves
    print("\nGenerating plots...")
    plot_learning_curves(experiments)

    print("\nAnalysis complete!")
