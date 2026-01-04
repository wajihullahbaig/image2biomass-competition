# log_and_plots.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_training_history(history, fold, session_dir):
    save_dir = os.path.join(session_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    
    def try_plot(ax_idx, key, label, color, style='-'):
        if key in history and len(history[key]) > 0:
            data = [float(x) for x in history[key] if x is not None]
            if len(data) > 0:
                axes[ax_idx].plot(data, label=label, color=color, linestyle=style)

    # 1. Total Loss
    try_plot(0, 'train_loss', 'Train', 'tab:blue')
    try_plot(0, 'val_loss', 'Val', 'tab:red')
    try_plot(0, 'holdout_loss', 'Holdout', 'tab:green')
    axes[0].set_title('Total Loss')

    # 2. Biomass Loss
    try_plot(1, 'train_bio', 'Train', 'tab:blue')
    try_plot(1, 'val_bio', 'Val', 'tab:red')
    try_plot(1, 'holdout_bio', 'Holdout', 'tab:green')
    axes[1].set_title('Biomass Loss')

    # 3. Aux Loss
    try_plot(2, 'train_aux', 'Train', 'tab:blue')
    try_plot(2, 'val_aux', 'Val', 'tab:red')
    try_plot(2, 'holdout_aux', 'Holdout', 'tab:green')
    axes[2].set_title('Aux Loss')

    # 4. Species Loss
    try_plot(3, 'train_sp', 'Train', 'tab:blue')
    try_plot(3, 'val_sp', 'Val', 'tab:red')
    try_plot(3, 'holdout_sp', 'Holdout', 'tab:green')
    axes[3].set_title('Species Loss')

    # 5. Physics Loss
    try_plot(4, 'train_phy', 'Train', 'tab:blue')
    try_plot(4, 'val_phy', 'Val', 'tab:red')
    try_plot(4, 'holdout_phy', 'Holdout', 'tab:green')
    axes[4].set_title('Physics Loss')

    # 6. R2 Metrics (UPDATED)
    try_plot(5, 'train_r2', 'Train R2', 'tab:blue')
    try_plot(5, 'val_r2', 'Val R2', 'tab:red')
    try_plot(5, 'holdout_r2', 'Holdout R2', 'tab:green')
    try_plot(5, 'score', 'Score', 'black', style='--')
    axes[5].set_title('R2 Metrics')
    axes[5].axhline(0, color='black', alpha=0.3)
    
    vals = []
    if 'val_r2' in history: vals.extend(history['val_r2'])
    if 'train_r2' in history: vals.extend(history['train_r2'])
    if 'holdout_r2' in history: vals.extend(history['holdout_r2'])
    
    if vals:
        vmin, vmax = min(vals), max(vals)
        axes[5].set_ylim(min(vmin - 0.1, -2.0), max(vmax + 0.1, 2.0))
    else:
        axes[5].set_ylim(-2.0, 2.0)

    # 7. Learning Rate
    try_plot(6, 'lr', 'Learning Rate', 'tab:purple')
    axes[6].set_title('Learning Rate')
    axes[6].set_yscale('log')

    # 8. Taxonomy Loss
    try_plot(7, 'train_tax', 'Train', 'tab:blue')
    try_plot(7, 'val_tax', 'Val', 'tab:red')
    try_plot(7, 'holdout_tax', 'Holdout', 'tab:green')
    axes[7].set_title('Taxonomy Loss')

    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_metrics.png"))
    plt.close()

    # --- NEW: Component-wise Plots ---
    comp_dir = os.path.join(save_dir, 'components')
    os.makedirs(comp_dir, exist_ok=True)

    # 1. Biomass Components
    fig_b, axes_b = plt.subplots(2, 3, figsize=(18, 10))
    axes_b = axes_b.flatten()
    
    bio_map = [
        ('loss_c', 'Clover Loss'), ('loss_d', 'Dead Loss'), ('loss_g', 'Green Loss'),
        ('loss_t', 'Total Loss'), ('loss_gdm', 'GDM Loss')
    ]
    
    for idx, (suffix, title) in enumerate(bio_map):
        try_plot_on_ax(axes_b[idx], history, f'train_{suffix}', f'val_{suffix}', title, holdout_key=f'holdout_{suffix}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f"fold_{fold}_biomass_components.png"))
    plt.close()

    # 2. Aux Components
    fig_a, axes_a = plt.subplots(1, 4, figsize=(24, 5))
    axes_a = axes_a.flatten()
    
    aux_map = [
        ('loss_ndvi', 'NDVI Loss'), ('loss_h', 'Height Loss'), 
        ('loss_int_mul', 'Interaction Mul Loss'), ('loss_int_add', 'Interaction Add Loss')
    ]
    
    for idx, (suffix, title) in enumerate(aux_map):
        try_plot_on_ax(axes_a[idx], history, f'train_{suffix}', f'val_{suffix}', title, holdout_key=f'holdout_{suffix}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f"fold_{fold}_aux_components.png"))
    plt.close()

def try_plot_on_ax(ax, history, train_key, val_key, title, holdout_key=None):
    """Helper to plot train/val curves on a given axis."""
    has_data = False
    if train_key in history and len(history[train_key]) > 0:
        ax.plot(history[train_key], label='Train', color='tab:blue')
        has_data = True
    if val_key in history and len(history[val_key]) > 0:
        ax.plot(history[val_key], label='Val', color='tab:red')
        has_data = True
    if holdout_key and holdout_key in history and len(history[holdout_key]) > 0:
        ax.plot(history[holdout_key], label='HO', color='tab:green', linestyle='--')
        has_data = True
        
    if has_data:
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False) # Hide empty plots

    
def setup_logging(logger_name="System Logger", log_dir='logs', file_name_part=None) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir_name = f"{file_name_part}_{timestamp}" if file_name_part else timestamp
    session_dir = os.path.join(log_dir, session_dir_name)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, 'plots'), exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    file_handler = logging.FileHandler(os.path.join(session_dir, 'session.log'), encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return session_dir

def log_fold_details(logger, train_df, val_df):
    def get_stats(df):
        if len(df) == 0: return "EMPTY", "EMPTY", "EMPTY"
        dates = f"{df['Sampling_Date'].min().date()} -> {df['Sampling_Date'].max().date()}"
        states = sorted(df['State'].unique().tolist())
        sp_counts = df['Species'].value_counts()
        species_str = str(sp_counts.head(5).to_dict())
        if len(sp_counts) > 10: species_str += "..."
        return dates, states, species_str

    t_dates, t_states, t_species = get_stats(train_df)
    v_dates, v_states, v_species = get_stats(val_df)
    
    msg = f"""
    \n    ----------------------------------------------------------------
    FOLD DETAILS
    ----------------------------------------------------------------
    [TRAIN] (n={len(train_df)})
      Dates:   {t_dates}
      States:  {t_states}
      Species: {t_species}
    ----------------------------------------------------------------
    [VALIDATION] (n={len(val_df)})
      Dates:   {v_dates}
      States:  {v_states}
      Species: {v_species}
    ----------------------------------------------------------------
    """
    logger.info(msg)

def log_upsample_stats(logger, before_df, after_df):
    if 'FunctionalGroup' in before_df.columns:
        col = 'FunctionalGroup'
    else:
        col = 'Species'
        
    sp_before = before_df[col].value_counts()
    sp_after = after_df[col].value_counts()
    
    all_cats = sorted(list(set(sp_before.index) | set(sp_after.index)))
    
    msg = "\n" + "="*60 + f"\nUPSAMPLING STATS ({col} Counts)\n" + "="*60
    msg += f"\nUpsampling based on column: {col}\n" 
    msg += f"\n{'Group':<20} | {'Before':<10} | {'After':<10} | {'Added':<10}"
    msg += "\n" + "-"*60
    
    for sp in all_cats:
        b = sp_before.get(sp, 0)
        a = sp_after.get(sp, 0)
        diff = a - b
        msg += f"\n{str(sp):<20} | {b:<10} | {a:<10} | +{diff:<10}"
        
    msg += "\n" + "="*60
    logger.info(msg)