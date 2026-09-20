# local_inference.py - Dual-Stream High-Resolution Inference with Soft Post-Processing
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add training module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..', 'training'))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from config.loader import cfg
from common import (
    TARGET_ORDER,
    soft_physics_postprocess,
    set_seed
)
from dataset import DualStreamBiomassDataset
from models import DualStreamBiomassModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_batch(model, img_l, img_r, use_tta=True):
    """
    Predicts biomass values for a batch of left and right views.
    Optionally applies horizontal flip TTA.
    """
    with torch.no_grad():
        if use_tta:
            reg1, _ = model(img_l, img_r)
            # TTA: horizontal flip of both sub-images
            img_l_flip = torch.flip(img_l, [3])
            img_r_flip = torch.flip(img_r, [3])
            reg2, _ = model(img_r_flip, img_l_flip)
            preds = [(r1 + r2) * 0.5 for r1, r2 in zip(reg1, reg2)]
        else:
            preds, _ = model(img_l, img_r)
            
        # Stack 5 targets: [B, 5]
        return torch.cat(preds, dim=1).cpu().numpy()


def run_inference(
    test_csv_path="test.csv",
    test_img_dir="test",
    model_checkpoint_dir=None,
    output_submission_path="submission.csv",
    use_tta=True,
    batch_size=8
):
    print("=" * 60)
    print("DUAL-STREAM DINO INFERENCE PIPELINE")
    print("=" * 60)
    set_seed(42)

    # 1. Load Test Metadata
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

    test_df_raw = pd.read_csv(test_csv_path)
    
    # Handle both long and wide format test.csv
    if 'target_name' in test_df_raw.columns:
        test_df_raw['clean_id'] = test_df_raw['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
        unique_samples = test_df_raw[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
    else:
        unique_samples = test_df_raw.copy()
        if 'clean_id' not in unique_samples.columns:
            unique_samples['clean_id'] = unique_samples['sample_id']
            
    print(f"Unique test images to process: {len(unique_samples)}")

    # 2. Find Models
    if model_checkpoint_dir is None:
        # Search in standard logs directory or user-named log folders (e.g. logs_kaggle_0.62)
        candidate_logs = sorted(glob.glob("logs/dual_stream_*") + glob.glob("logs_*/dual_stream_*"), reverse=True)
        if candidate_logs:
            model_checkpoint_dir = candidate_logs[0]
        else:
            raise FileNotFoundError("No trained dual_stream checkpoints found in logs/ or logs_*/!")
            
    model_paths = sorted(glob.glob(os.path.join(model_checkpoint_dir, "best_model_fold*.pt")))
    if not model_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {model_checkpoint_dir}")
        
    print(f"Found {len(model_paths)} checkpoints in: {model_checkpoint_dir}")

    # 3. Create Dataset and DataLoader
    img_size = cfg.preprocessing.image_height
    test_dataset = DualStreamBiomassDataset(
        unique_samples,
        img_dir=test_img_dir,
        img_size=img_size,
        is_training=False,
        camera_scaling_prob=0.0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 4. Predict across Ensembled Folds
    all_fold_preds = []
    
    for fold_idx, model_path in enumerate(model_paths):
        print(f"Loading Fold {fold_idx + 1}: {os.path.basename(model_path)}...")
        model = DualStreamBiomassModel(
            backbone_name=cfg.hyperparameters.backbone,
            num_targets=5,
            num_intervals=cfg.loss.num_intervals,
            fusion_dim=cfg.training.fusion_dim,
            pretrained=False
        ).to(DEVICE)
        
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        fold_preds = []
        for batch in tqdm(test_loader, desc=f"Inference Fold {fold_idx + 1}", leave=False):
            img_l = batch['image_left'].to(DEVICE)
            img_r = batch['image_right'].to(DEVICE)
            preds_batch = predict_batch(model, img_l, img_r, use_tta=use_tta)
            fold_preds.append(preds_batch)
            
        all_fold_preds.append(np.concatenate(fold_preds, axis=0))

    # 5. Average Predictions across Folds
    avg_preds_raw = np.mean(all_fold_preds, axis=0)

    # 6. Apply Soft Physical Post-Processing & Calibration
    avg_preds_post = soft_physics_postprocess(avg_preds_raw)

    # 7. Build Kaggle Submission
    clean_ids = unique_samples['clean_id'].tolist() if 'clean_id' in unique_samples.columns else unique_samples['sample_id'].tolist()
    pred_dict = {
        clean_ids[i]: {col: avg_preds_post[i, c_idx] for c_idx, col in enumerate(TARGET_ORDER)}
        for i in range(len(clean_ids))
    }

    if 'target_name' in test_df_raw.columns:
        # Fill existing long-format dataframe
        submission_df = test_df_raw.copy()
        submission_df['target'] = submission_df.apply(
            lambda row: pred_dict.get(row['clean_id'], {}).get(row['target_name'], 0.0),
            axis=1
        )
        final_submission = submission_df[['sample_id', 'target']]
    else:
        # Generate long format from wide IDs
        records = []
        for cid in clean_ids:
            for col in TARGET_ORDER:
                records.append({
                    'sample_id': f"{cid}__{col}",
                    'target': pred_dict[cid][col]
                })
        final_submission = pd.DataFrame(records)

    final_submission.to_csv(output_submission_path, index=False)
    print(f"\n[OK] Submission successfully saved to: {output_submission_path}")
    print(f"Sample preview:\n{final_submission.head(10)}")
    return final_submission


if __name__ == "__main__":
    run_inference()