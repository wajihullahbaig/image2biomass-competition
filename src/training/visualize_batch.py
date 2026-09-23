import os
import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def unnormalize(tensor):
    """Converts a [3, H, W] normalized tensor back to [H, W, 3] uint8/float RGB."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)

def save_sample_batch(batch, save_path, max_samples=4, title_prefix="Train Batch"):
    """
    Saves a visual grid showing left/right views and target annotations for a batch.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imgs_l = batch['image_left']
    imgs_r = batch['image_right']
    targets = batch.get('targets', None)
    sample_ids = batch.get('sample_id', [f"id_{i}" for i in range(len(imgs_l))])
    
    n = min(len(imgs_l), max_samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(n):
        rgb_l = unnormalize(imgs_l[i])
        rgb_r = unnormalize(imgs_r[i])
        
        sid = sample_ids[i] if isinstance(sample_ids[i], str) else str(sample_ids[i])
        target_str = ""
        if targets is not None:
            t = targets[i].cpu().numpy()
            if len(t) >= 3:
                target_str = f"Green: {t[0]:.1f}g | Dead: {t[1]:.1f}g | Clover: {t[2]:.1f}g"
            else:
                target_str = f"Targets: {t}"
                
        axes[i, 0].imshow(rgb_l)
        axes[i, 0].set_title(f"[{title_prefix}] {sid} - Left View\n{target_str}", fontsize=9)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(rgb_r)
        axes[i, 1].set_title(f"[{title_prefix}] {sid} - Right View", fontsize=9)
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample batch visualization: {save_path}")
