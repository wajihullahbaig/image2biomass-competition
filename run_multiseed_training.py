"""
Multi-Seed Ensemble Training Script

This script trains multiple models with different random seeds to create
a diverse ensemble. The models can then be averaged during inference for
better generalization and robustness to label noise.

Usage:
    python run_multiseed_training.py

This will train 3 models with seeds: 42, 123, 456
"""

import subprocess
import yaml
from pathlib import Path

# Configuration
SEEDS = [42, 123, 456]
CONFIG_PATH = Path("src/training/config/config.yaml")

def update_seed_in_config(seed):
    """Update the random_seed in config.yaml"""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    config['hyperparameters']['random_seed'] = seed
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated config with seed: {seed}")

def run_training(seed_idx, seed):
    """Run training with the specified seed"""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL {seed_idx + 1}/{ len(SEEDS)} WITH SEED {seed}")
    print(f"{'='*80}\n")
    
    # Update config
    update_seed_in_config(seed)
    
    # Run training
    result = subprocess.run(
        ["python", "src/training/train_unified_holdout.py"],
        cwd=".",
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed for seed {seed}")
        return False
    
    print(f"✓ Completed training for seed {seed}")
    return True

def main():
    print(f"\n{'='*80}")
    print(f"MULTI-SEED ENSEMBLE TRAINING")
    print(f"Training {len(SEEDS)} models with seeds: {SEEDS}")
    print(f"{'='*80}\n")
    
    successful_seeds = []
    failed_seeds = []
    
    for idx, seed in enumerate(SEEDS):
        success = run_training(idx, seed)
        if success:
            successful_seeds.append(seed)
        else:
            failed_seeds.append(seed)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"MULTI-SEED TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {len(successful_seeds)}/{len(SEEDS)} models")
    if successful_seeds:
        print(f"  Seeds: {successful_seeds}")
    if failed_seeds:
        print(f"❌ Failed: {len(failed_seeds)} models")
        print(f"  Seeds: {failed_seeds}")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("1. Update inference scripts to load all 3 seed models")
    print("2. Average their predictions for final submission")
    print("3. Expected improvement: +0.02-0.04 on LB")

if __name__ == "__main__":
    main()
