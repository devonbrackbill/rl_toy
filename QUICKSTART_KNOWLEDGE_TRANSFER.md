# Quick Start: Knowledge Transfer Experiments

This guide shows you how to run knowledge transfer experiments to study how much imitation data is required for rapid learning.

## Prerequisites

You should have already:
1. Installed dependencies: `pip install -r requirements.txt`
2. Trained a model: `python train.py` (creates `save.p`)

## Step-by-Step Guide

### Step 1: Generate Synthetic Demonstrations (5-10 minutes)

Generate demonstration data from your trained model:

```bash
python generate_demonstrations.py
```

**What happens:**
- Loads your trained model from `save.p`
- Plays 10 episodes of Pong
- Saves data to `model_demonstrations/model_demo_10ep_<timestamp>.npz`
- Shows statistics: episode rewards, action distribution, win rate

**Expected output:**
```
Loading trained model from save.p...
Model loaded. W1 shape: (200, 6400), W2 shape: (200,)

Generating 10 demonstration episodes...
------------------------------------------------------------
Episode 1/10: Steps=1234, Reward=3, Action distribution: UP=567, DOWN=667
Episode 2/10: Steps=1089, Reward=5, Action distribution: UP=512, DOWN=577
...
------------------------------------------------------------

Generation complete!
Total timesteps: 12456
Average episode reward: 4.20
Win rate: 80.0%
```

**Optional:** Edit `generate_demonstrations.py` to change `num_episodes = 10` to generate more data.

---

### Step 2: Run Knowledge Transfer Experiments (1-2 hours)

Train new models with different amounts of demonstration data:

```bash
# Quick test with fewer imitation episodes (faster)
python train_from_model_demos.py --imitation-episodes 200 --fractions 0.1 0.5 1.0

# Full experiment with default settings
python train_from_model_demos.py
```

**What happens:**
- Trains 5 separate models (10%, 25%, 50%, 75%, 100% of data)
- Each model trains for 1000 imitation episodes
- Evaluates performance every 10 episodes
- Saves models to `experiments/data_XX_pct_after_imitation.p`
- Saves metrics to `experiments/data_XX_pct_results.npz`

**Expected output:**
```
======================================================================
Experiment: data_10_pct
Data fraction: 10.0%
======================================================================
Loading model demonstrations from: model_demonstrations/model_demo_10ep_...npz
Using 10.0% of data: 1246/12456 samples
Action distribution: UP=556, DOWN=690

Evaluating initial random policy...
Initial policy: avg_reward=-18.33, avg_steps=1123.7

Starting imitation learning for 1000 episodes...
----------------------------------------------------------------------
Episode 10/1000 | Loss: 0.6234 | Acc: 0.621 | Eval reward: -12.33 | Steps: 1056.3
Episode 20/1000 | Loss: 0.5891 | Acc: 0.689 | Eval reward: -8.67 | Steps: 989.7
...
```

**Command options:**
```bash
# Experiment with specific data fractions
python train_from_model_demos.py --fractions 0.05 0.1 0.25 0.5

# Change number of imitation training episodes
python train_from_model_demos.py --imitation-episodes 500

# Add RL episodes after imitation (to see continued improvement)
python train_from_model_demos.py --rl-episodes 100
```

---

### Step 3: Analyze Results (1 minute)

Compare learning efficiency across experiments:

```bash
python analyze_results.py
```

**What happens:**
- Loads all experiment results from `experiments/`
- Prints summary table with final metrics
- Generates 4 comparison plots
- Saves visualization to `experiments/learning_curves_comparison.png`

**Expected output:**
```
Loading experiment results...
Found 5 experiments

================================================================================
EXPERIMENT RESULTS SUMMARY
================================================================================
Data %     Episodes     Final Loss   Final Acc    Final Reward    Best Reward
--------------------------------------------------------------------------------
10.0       1000         0.4123       0.812        -2.33           1.67
25.0       1000         0.3856       0.841        3.67            5.33
50.0       1000         0.3234       0.879        8.33            10.00
75.0       1000         0.3012       0.892        11.67           13.33
100.0      1000         0.2891       0.901        14.33           16.00
================================================================================

================================================================================
DATA EFFICIENCY ANALYSIS
================================================================================

To reach avg reward >= 0:
   25.0% data: reached at episode 340
   50.0% data: reached at episode 180
   75.0% data: reached at episode 120
  100.0% data: reached at episode 90

To reach avg reward >= 5:
   50.0% data: reached at episode 520
   75.0% data: reached at episode 340
  100.0% data: reached at episode 210
...
```

---

## Understanding the Results

### Key Metrics

1. **Loss**: Binary cross-entropy loss (lower is better)
   - Measures how well the model predicts the teacher's actions
   - Should decrease over time

2. **Accuracy**: Action prediction accuracy (higher is better)
   - Percentage of actions matching the teacher
   - Should increase over time

3. **Eval Reward**: Average game reward (higher is better)
   - Measures actual game performance
   - Positive = winning, negative = losing

### What to Look For

1. **Minimum viable data**:
   - At what data fraction does the model start learning effectively?
   - Is there a threshold below which learning fails?

2. **Diminishing returns**:
   - Does 100% data perform much better than 50%?
   - Where do you get the best data efficiency?

3. **Learning speed**:
   - Do models with more data learn faster (reach threshold in fewer episodes)?
   - Or do they just reach a higher final performance?

4. **Performance ceiling**:
   - What's the maximum performance achieved?
   - How close do models with less data get to this ceiling?

---

## Experiment Variations

### Generate More Demonstrations

Edit `generate_demonstrations.py`:
```python
num_episodes = 50  # Generate more episodes for more data
```

Then re-run step 1.

### Test Different Data Amounts

Focus on specific ranges:
```bash
# Fine-grained low-data regime
python train_from_model_demos.py --fractions 0.01 0.02 0.05 0.1 0.2

# High-data regime only
python train_from_model_demos.py --fractions 0.5 0.6 0.7 0.8 0.9 1.0
```

### Shorter Experiments for Quick Tests

For rapid iteration:
```bash
python train_from_model_demos.py --imitation-episodes 100 --fractions 0.1 1.0
```

### Include RL Training

See if RL can compensate for less imitation data:
```bash
python train_from_model_demos.py --rl-episodes 100 --fractions 0.1 0.5 1.0
```

---

## Expected Timeline

- **Generate demos (10 episodes)**: 5-10 minutes
- **Train 1 model (1000 imitation episodes)**: ~15-20 minutes
- **Full experiment (5 models)**: 1-2 hours
- **Analysis**: < 1 minute

**Total time for complete experiment**: ~2 hours

---

## Troubleshooting

### "No model demonstration files found"
- Make sure you ran `generate_demonstrations.py` first
- Check that `model_demonstrations/` directory exists with `.npz` files

### "No experiment results found"
- Make sure you ran `train_from_model_demos.py` first
- Check that `experiments/` directory has `*_results.npz` files

### Training is too slow
- Reduce `--imitation-episodes` (e.g., 200 instead of 1000)
- Test with fewer data fractions (e.g., just 0.1 and 1.0)

### Out of memory
- Reduce `num_episodes` in `generate_demonstrations.py`
- This will reduce total data size

---

## Next Steps

After analyzing results:

1. **Tune hyperparameters**: Try different learning rates in `train_from_model_demos.py`
2. **Test transfer robustness**: Generate demos from different checkpoints
3. **Cross-validation**: Run multiple seeds and average results
4. **Compare to human data**: Run same experiments with human demonstrations

---

## Questions to Explore

- How does model performance (teacher quality) affect data efficiency?
- Is there a "sweet spot" data amount for fastest learning?
- Can models trained on 10% data + RL surpass models trained on 100% data?
- How much does data quality matter vs. data quantity?
