# log_and_plots.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_training_history(history, fold, session_dir):
    """
    Plots metrics including component-wise losses for Train/Val.
    Updated to include Taxonomy Loss.
    """
    save_dir = os.path.join(session_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # 2x4 Grid to accommodate 8 plots (Bio, Aux, Sp, Phy, Tax, Total, R2, LR)
    fig, axes = plt.subplots(3, 3, figsize=(24, 10))
    axes = axes.flatten()
    
    def try_plot(ax_idx, key, label, color, style='-'):
        if key in history and len(history[key]) > 0:
            # Defensive conversion to float to ensure matplotlib compatibility
            data = [float(x) for x in history[key] if x is not None]
            if len(data) > 0:
                axes[ax_idx].plot(data, label=label, color=color, linestyle=style)

    # 1. Total Loss
    try_plot(0, 'train_loss', 'Train', 'tab:blue')
    try_plot(0, 'val_loss', 'Val', 'tab:red')
    axes[0].set_title('Total Loss')

    # 2. Biomass Loss
    try_plot(1, 'train_bio', 'Train', 'tab:blue')
    try_plot(1, 'val_bio', 'Val', 'tab:red')
    axes[1].set_title('Biomass Loss')

    # 3. Aux Loss
    try_plot(2, 'train_aux', 'Train', 'tab:blue')
    try_plot(2, 'val_aux', 'Val', 'tab:red')
    axes[2].set_title('Aux Loss')

    # 4. Species Loss
    try_plot(3, 'train_sp', 'Train', 'tab:blue')
    try_plot(3, 'val_sp', 'Val', 'tab:red')
    axes[3].set_title('Species Loss')
   
    # 5. Taxonomy Loss (NEW)
    try_plot(5, 'train_tax', 'Train', 'tab:blue')
    try_plot(5, 'val_tax', 'Val', 'tab:red')
    axes[5].set_title('Taxonomy Loss')

    # 6. Physics Loss
    try_plot(6, 'train_phy', 'Train', 'tab:blue')
    try_plot(6, 'val_phy', 'Val', 'tab:red')
    axes[6].set_title('Physics Loss')

    # 7. R2 Metrics
    try_plot(7, 'val_r2', 'Val R2', 'red')
    axes[7].set_title('R2 Metrics')
    axes[7].axhline(0, color='black', alpha=0.3)
    
    # Flexible ylim for R2
    vals = []
    if 'val_r2' in history: vals.extend(history['val_r2'])
    if vals:
        vmin, vmax = min(vals), max(vals)
        axes[7].set_ylim(min(vmin - 0.1, -1.5), max(vmax + 0.1, 1.5))
    else:
        axes[7].set_ylim(-1.5,1.5)

    # 8. Learning Rate
    try_plot(8, 'lr', 'Learning Rate', 'tab:purple')
    axes[8].set_title('Learning Rate')
    axes[8].set_yscale('log')

    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_metrics.png"))
    plt.close()

    
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
        # Top 5 most common species for brevity if list is long
        sp_counts = df['Species'].value_counts()
        species_str = str(sp_counts.head(5).to_dict())
        if len(sp_counts) > 5: species_str += "..."
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
    msg += f"\n{'Group':<20} | {'Before':<10} | {'After':<10} | {'Added':<10}"
    msg += "\n" + "-"*60
    
    for sp in all_cats:
        b = sp_before.get(sp, 0)
        a = sp_after.get(sp, 0)
        diff = a - b
        msg += f"\n{str(sp):<20} | {b:<10} | {a:<10} | +{diff:<10}"
        
    msg += "\n" + "="*60
    logger.info(msg)