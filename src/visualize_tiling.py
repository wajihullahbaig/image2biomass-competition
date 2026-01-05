# visualize_tiling.py - Visualize Tile Augmentation Strategies
"""
This script generates visual comparisons of different augmentation modes.
Useful for understanding and debugging the tiling system.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

def create_demo_image(width=512, height=256):
    """
    Create a synthetic grass-like demo image for visualization.
    Simulates a real quadrat with spatial variation.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    # Create gradient + noise pattern (simulates grass texture)
    np.random.seed(42)
    for x in range(width):
        for y in range(height):
            # Base green color with spatial variation
            base_green = 120 + 30 * (y / height)
            noise = np.random.randint(-20, 20)
            
            r = int(np.clip(50 + noise, 0, 255))
            g = int(np.clip(base_green + noise, 0, 255))
            b = int(np.clip(30 + noise, 0, 255))
            
            pixels[x, y] = (r, g, b)
    
    # Add some "patches" to simulate clover/dead grass
    for _ in range(10):
        patch_x = np.random.randint(0, width - 50)
        patch_y = np.random.randint(0, height - 50)
        colors = [(80, 40, 20), (180, 220, 100)]
        patch_color = colors[np.random.randint(len(colors))]
        
        for dx in range(50):
            for dy in range(50):
                if np.random.rand() > 0.3:  # Irregular shape
                    x, y = patch_x + dx, patch_y + dy
                    if 0 <= x < width and 0 <= y < height:
                        pixels[x, y] = patch_color
    
    return img

def split_into_tiles(image):
    """Split image into 4 tiles (2x2 grid)."""
    w, h = image.size
    mid_w, mid_h = w // 2, h // 2
    
    tiles = [
        image.crop((0, 0, mid_w, mid_h)),           # TL
        image.crop((mid_w, 0, w, mid_h)),           # TR
        image.crop((0, mid_h, mid_w, h)),           # BL
        image.crop((mid_w, mid_h, w, h))            # BR
    ]
    return tiles

def stitch_tiles(tiles):
    """Reconstruct image from tiles."""
    tile_w, tile_h = tiles[0].size
    stitched = Image.new('RGB', (tile_w * 2, tile_h * 2))
    
    stitched.paste(tiles[0], (0, 0))
    stitched.paste(tiles[1], (tile_w, 0))
    stitched.paste(tiles[2], (0, tile_h))
    stitched.paste(tiles[3], (tile_w, tile_h))
    
    return stitched

def apply_tile_transforms(tiles, seed=None):
    """Apply random flips to tiles."""
    if seed is not None:
        np.random.seed(seed)
    
    transformed = []
    for i, tile in enumerate(tiles):
        # Random flips
        if np.random.rand() > 0.5:
            tile = TF.hflip(tile)
        if np.random.rand() > 0.5:
            tile = TF.vflip(tile)
        transformed.append(tile)
    
    return transformed

def visualize_augmentation_modes(image, output_path='augmentation_comparison.png'):
    """
    Create comprehensive visualization of all augmentation modes.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Tile-Based Augmentation Strategy Visualization', fontsize=16, fontweight='bold')
    
    # Row 1: Original and Tile Extraction
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image\n(Full Biomass: 100g)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    tiles = split_into_tiles(image)
    tile_labels = ['Top-Left (25g)', 'Top-Right (25g)', 'Bottom-Left (25g)', 'Bottom-Right (25g)']
    
    for i, (tile, label) in enumerate(zip(tiles, tile_labels)):
        axes[0, i+1].imshow(tile) if i < 3 else None
        if i < 3:
            axes[0, i+1].set_title(f'Tile {i}\n{label}', fontsize=10)
            axes[0, i+1].axis('off')
    
    # Remove unused subplot
    axes[0, 3].axis('off')
    
    # Row 2: STITCH Mode (Texture Variation)
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original\n(Before Stitch)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Generate 3 different stitched variations
    for i in range(3):
        tiles_copy = split_into_tiles(image)
        tiles_transformed = apply_tile_transforms(tiles_copy, seed=i)
        stitched = stitch_tiles(tiles_transformed)
        
        axes[1, i+1].imshow(stitched)
        axes[1, i+1].set_title(f'Stitched Variant {i+1}\n(Full Biomass: 100g)', fontsize=10)
        axes[1, i+1].axis('off')
        
        # Add annotation
        if i == 0:
            axes[1, i+1].text(0.5, -0.1, 'Random H/V flips applied per tile', 
                            transform=axes[1, i+1].transAxes, 
                            ha='center', fontsize=9, style='italic')
    
    # Row 3: DIVIDE Mode (Density Learning)
    axes[2, 0].imshow(image)
    axes[2, 0].set_title('Original\n(Before Division)', fontsize=12)
    axes[2, 0].axis('off')
    
    tiles = split_into_tiles(image)
    tiles = apply_tile_transforms(tiles, seed=42)
    
    for i, tile in enumerate(tiles):
        axes[2, i+1].imshow(tile) if i < 3 else axes[2, 3].imshow(tile)
        axes[2, i+1 if i < 3 else 3].set_title(f'Independent Sample {i+1}\n(Biomass: 25g)', fontsize=10)
        axes[2, i+1 if i < 3 else 3].axis('off')
        
        if i == 0:
            axes[2, i+1].text(0.5, -0.1, '4 separate training samples, each with targets/4', 
                            transform=axes[2, i+1].transAxes, 
                            ha='center', fontsize=9, style='italic')
    
    # Add row labels
    row_labels = ['Mode 0: Original\n(No Augmentation)', 
                  'Mode 1: STITCH\n(Texture Variation)', 
                  'Mode 2: DIVIDE\n(Density Learning)']
    
    for i, label in enumerate(row_labels):
        fig.text(0.05, 0.83 - i*0.28, label, fontsize=12, fontweight='bold', 
                ha='center', va='center', rotation=90,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0.08, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()

def visualize_training_pipeline(image, output_path='training_pipeline.png'):
    """
    Visualize how one image becomes 6 training samples.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Training Data Expansion: 1 Image → 6 Training Samples', 
                fontsize=16, fontweight='bold')
    
    # Original image (centered top)
    ax_orig = fig.add_subplot(gs[0, 1])
    ax_orig.imshow(image)
    ax_orig.set_title('1 Original Image\n(100g total biomass)', fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    # Arrow annotation
    fig.text(0.5, 0.65, '↓ Augmentation ↓', ha='center', fontsize=12, 
            fontweight='bold', style='italic')
    
    # Row 2: Sample 1 (Original) + Sample 2 (Stitched)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(image)
    ax1.set_title('Sample 1: Original\n(100g)', fontsize=11, color='green', fontweight='bold')
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes, 
                                fill=False, edgecolor='green', linewidth=3))
    
    tiles = split_into_tiles(image)
    tiles = apply_tile_transforms(tiles, seed=1)
    stitched = stitch_tiles(tiles)
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(stitched)
    ax2.set_title('Sample 2: Stitched\n(100g)', fontsize=11, color='blue', fontweight='bold')
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, 
                                fill=False, edgecolor='blue', linewidth=3))
    
    # Row 2-3: Samples 3-6 (Divided Tiles)
    tiles = split_into_tiles(image)
    tiles = apply_tile_transforms(tiles, seed=42)
    
    positions = [(1, 2), (2, 0), (2, 1), (2, 2)]
    colors = ['red', 'orange', 'purple', 'brown']
    
    for i, (tile, pos, color) in enumerate(zip(tiles, positions, colors)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.imshow(tile)
        ax.set_title(f'Sample {i+3}: Tile {i+1}\n(25g)', fontsize=11, 
                    color=color, fontweight='bold')
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                   fill=False, edgecolor=color, linewidth=3))
    
    # Summary box
    summary_text = """
    🎯 Result: 6 Training Samples from 1 Image
    
    ✓ Sample 1: Original (100g) - Baseline
    ✓ Sample 2: Stitched (100g) - Texture variation
    ✓ Samples 3-6: Tiles (25g each) - Density learning
    
    📊 Effective Data Size: 357 → 2,142 samples (6× increase)
    """
    
    fig.text(0.5, 0.05, summary_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Pipeline visualization saved to: {output_path}")
    plt.close()

def visualize_target_scaling(output_path='target_scaling.png'):
    """
    Visualize how targets are scaled for different augmentation modes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Target Scaling Strategy', fontsize=16, fontweight='bold')
    
    # Original targets
    targets = {
        'Clover': 20,
        'Dead': 10,
        'Green': 30,
        'Total': 60,
        'GDM': 50
    }
    
    # Mode 0: Original
    ax = axes[0]
    bars = ax.bar(targets.keys(), targets.values(), color=['#8B4513', '#A0522D', '#228B22', '#4169E1', '#FFD700'])
    ax.set_title('Mode 0: Original\n(Scale: 1.0)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Biomass (grams)', fontsize=12)
    ax.set_ylim(0, 70)
    
    # Add value labels
    for bar, val in zip(bars, targets.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}g', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Mode 1: Stitched (Same as original)
    ax = axes[1]
    bars = ax.bar(targets.keys(), targets.values(), color=['#8B4513', '#A0522D', '#228B22', '#4169E1', '#FFD700'])
    ax.set_title('Mode 1: Stitched\n(Scale: 1.0)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Biomass (grams)', fontsize=12)
    ax.set_ylim(0, 70)
    
    for bar, val in zip(bars, targets.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}g', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Mode 2: Divided Tiles
    scaled_targets = {k: v * 0.25 for k, v in targets.items()}
    ax = axes[2]
    bars = ax.bar(scaled_targets.keys(), scaled_targets.values(), 
                 color=['#8B4513', '#A0522D', '#228B22', '#4169E1', '#FFD700'])
    ax.set_title('Mode 2: Divided Tiles\n(Scale: 0.25)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Biomass (grams)', fontsize=12)
    ax.set_ylim(0, 70)
    
    for bar, val in zip(bars, scaled_targets.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}g', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add explanation
    explanation = """
    ℹ️ Target Scaling Logic:
    • Original & Stitched: Full biomass (spatially uniform grass)
    • Divided Tiles: Each tile has exactly 1/4 of total (valid due to spatial uniformity)
    • Physics constraint maintained: GDM = Clover + Green (even after scaling)
    """
    fig.text(0.5, -0.05, explanation, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Target scaling visualization saved to: {output_path}")
    plt.close()

def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("TILE AUGMENTATION VISUALIZATION GENERATOR")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = 'tiling_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate demo image
    print("📸 Generating demo grass quadrat image...")
    demo_image = create_demo_image(width=512, height=256)
    demo_path = os.path.join(output_dir, 'demo_quadrat.png')
    demo_image.save(demo_path)
    print(f"✅ Demo image saved to: {demo_path}")
    
    # Generate visualizations
    print("\n🎨 Generating augmentation comparison...")
    visualize_augmentation_modes(
        demo_image, 
        output_path=os.path.join(output_dir, 'augmentation_comparison.png')
    )
    
    print("\n📊 Generating training pipeline visualization...")
    visualize_training_pipeline(
        demo_image,
        output_path=os.path.join(output_dir, 'training_pipeline.png')
    )
    
    print("\n📈 Generating target scaling visualization...")
    visualize_target_scaling(
        output_path=os.path.join(output_dir, 'target_scaling.png')
    )
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"📁 Output directory: {output_dir}/")
    print("="*70 + "\n")
    
    print("Generated files:")
    print(f"  1. demo_quadrat.png - Synthetic grass quadrat")
    print(f"  2. augmentation_comparison.png - Side-by-side comparison of modes")
    print(f"  3. training_pipeline.png - 1 image → 6 samples expansion")
    print(f"  4. target_scaling.png - Target value scaling visualization")
    print("\n💡 Review these images to understand the augmentation strategy!")

if __name__ == '__main__':
    main()