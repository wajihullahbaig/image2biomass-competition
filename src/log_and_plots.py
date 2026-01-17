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
    
    # Hide the last axes (we only have 7 plots now)
    axes[7].set_visible(False)
    
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

    # 5. Derived Losses (Total/GDM) - Updated for new naming
    # Total losses
    if 'train_loss_total' in history and len(history['train_loss_total']) > 0:
        axes[4].plot([float(x) for x in history['train_loss_total']], label='Train Total', color='tab:blue')
    if 'val_loss_total' in history and len(history['val_loss_total']) > 0:
        axes[4].plot([float(x) for x in history['val_loss_total']], label='Val Total', color='tab:red')
    if 'holdout_loss_total' in history and len(history['holdout_loss_total']) > 0:
        axes[4].plot([float(x) for x in history['holdout_loss_total']], label='HO Total', color='tab:green')
    
    # GDM losses  
    if 'train_loss_gdm' in history and len(history['train_loss_gdm']) > 0:
        axes[4].plot([float(x) for x in history['train_loss_gdm']], label='Train GDM', color='tab:orange', linestyle='--')
    if 'val_loss_gdm' in history and len(history['val_loss_gdm']) > 0:
        axes[4].plot([float(x) for x in history['val_loss_gdm']], label='Val GDM', color='tab:pink', linestyle='--')
    if 'holdout_loss_gdm' in history and len(history['holdout_loss_gdm']) > 0:
        axes[4].plot([float(x) for x in history['holdout_loss_gdm']], label='HO GDM', color='tab:olive', linestyle='--')
    
    # Green losses
    if 'train_loss_green' in history and len(history['train_loss_green']) > 0:
        axes[4].plot([float(x) for x in history['train_loss_green']], label='Train Green', color='tab:cyan', linestyle=':')
    if 'val_loss_green' in history and len(history['val_loss_green']) > 0:
        axes[4].plot([float(x) for x in history['val_loss_green']], label='Val Green', color='tab:brown', linestyle=':')
    if 'holdout_loss_green' in history and len(history['holdout_loss_green']) > 0:
        axes[4].plot([float(x) for x in history['holdout_loss_green']], label='HO Green', color='tab:gray', linestyle=':')
    
    axes[4].set_title('Primary Losses (Total, GDM, Green)')

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


    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_metrics.png"))
    plt.close()

    comp_dir = os.path.join(save_dir, 'components')
    os.makedirs(comp_dir, exist_ok=True)

    # 1. Biomass Components
    fig_b, axes_b = plt.subplots(2, 3, figsize=(18, 10))
    axes_b = axes_b.flatten()
    
    bio_map = [
        ('loss_clover', 'Clover Loss'), ('loss_dead', 'Dead Loss'), ('loss_green', 'Green Loss'),
        ('loss_total', 'Total Loss'), ('loss_gdm', 'GDM Loss')
    ]
    
    for idx, (suffix, title) in enumerate(bio_map):
        try_plot_on_ax(axes_b[idx], history, f'train_{suffix}', f'val_{suffix}', title, holdout_key=f'holdout_{suffix}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f"fold_{fold}_biomass_components.png"))
    plt.close()

    # 2. Aux Components
    fig_a, axes_a = plt.subplots(2, 3, figsize=(18, 10))
    axes_a = axes_a.flatten()
    
    aux_map = [
        ('loss_ndvi', 'NDVI Loss'), ('loss_h', 'Height Loss'), 
        ('loss_int_mul', 'Interaction Mul Loss'), ('loss_int_add', 'Interaction Add Loss'),
        ('loss_hsv', 'HSV Green Loss')
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

def get_basic_stats(df):
        if len(df) == 0: return "EMPTY", "EMPTY"
        dates = f"{df['Sampling_Date'].min().date()} -> {df['Sampling_Date'].max().date()}"
        states = sorted(df['State'].unique().tolist())
        return dates, states
    
def log_dataframe_details(logger, df, name="DataFrame"):
    t_dates, t_states = get_basic_stats(df)
    # Species Table
    sp_counts = df['Species'].value_counts()
    all_species = sorted(sp_counts.index.tolist())
    table_msg = f"{'Species':<30} | {'Count':>8}"
    table_msg += "\n    " + "-" * 40
    total_count = 0
    for sp in all_species:
        count = sp_counts.get(sp, 0)
        table_msg += f"\n    {str(sp)[:30]:<30} | {count:>8}"
        total_count += count
    table_msg += "\n    " + "-" * 40
    table_msg += f"\n    {'TOTAL':<30} | {total_count:>8}"
    msg = f"""
    \n    -----------------------------------------------------------------
    {name} DETAILS
    -----------------------------------------------------------------
    Dates:  {t_dates}
    States: {t_states}
    SPECIES DISTRIBUTION:
    {table_msg}
    -----------------------------------------------------------------
    """
    logger.info(msg)
    
    
def log_fold_details(logger, train_df, val_df):
    

    t_dates, t_states = get_basic_stats(train_df)
    v_dates, v_states = get_basic_stats(val_df)
    
    # Species Table
    sp_train = train_df['Species'].value_counts()
    sp_val = val_df['Species'].value_counts()
    all_species = sorted(list(set(sp_train.index) | set(sp_val.index)))
    
    table_msg = f"{'Species':<30} | {'Train':>8} | {'Val':>8} | {'Total':>8}"
    table_msg += "\n    " + "-" * 61
    
    total_train = 0
    total_val = 0
    
    for sp in all_species:
        tc = sp_train.get(sp, 0)
        vc = sp_val.get(sp, 0)
        tot = tc + vc
        table_msg += f"\n    {str(sp)[:30]:<30} | {tc:>8} | {vc:>8} | {tot:>8}"
        total_train += tc
        total_val += vc
        
    table_msg += "\n    " + "-" * 61
    table_msg += f"\n    {'TOTAL':<30} | {total_train:>8} | {total_val:>8} | {total_train + total_val:>8}"

    msg = f"""
    \n    -----------------------------------------------------------------
    FOLD DETAILS
    -----------------------------------------------------------------
    [TRAIN] ({total_train} samples)
      Dates:  {t_dates}
      States: {t_states}
    
    [VALIDATION] ({total_val} samples)
      Dates:  {v_dates}
      States: {v_states}
    
    SPECIES DISTRIBUTION:
    {table_msg}
    -----------------------------------------------------------------
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
    
    
def get_formatted_loss_log(epoch, train_metrics, val_metrics, hol_metrics, current_score, score_gap, lr, v_r2, h_r2):
    """
    Generate a formatted ASCII table for training, validation, and holdout losses and metrics.
    """
    def fmt_num(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "-"
    def fmt_sci(v):
        try:
            return f"{float(v):.4e}"
        except (TypeError, ValueError):
            return "-"

    rows = [
        ("Loss (Total)", fmt_num(train_metrics.get('train_loss')), fmt_num(val_metrics.get('val_loss')), fmt_num(hol_metrics.get('holdout_loss'))),
        ("Biomass",      fmt_num(train_metrics.get('train_bio')),  fmt_num(val_metrics.get('val_bio')),  fmt_num(hol_metrics.get('holdout_bio'))),
        ("Aux",          fmt_num(train_metrics.get('train_aux')),  fmt_num(val_metrics.get('val_aux')),  fmt_num(hol_metrics.get('holdout_aux'))),
        ("Species",      fmt_num(train_metrics.get('train_sp')),   fmt_num(val_metrics.get('val_sp')),   fmt_num(hol_metrics.get('holdout_sp'))),
        ("HSV",          fmt_num(train_metrics.get('train_loss_hsv')), fmt_num(val_metrics.get('val_loss_hsv')), fmt_num(hol_metrics.get('holdout_loss_hsv'))),
        ("R2",           fmt_num(train_metrics.get('train_r2')),   fmt_num(v_r2),                         fmt_num(h_r2)),
    ]

    # Compute column widths
    w0 = max(len("Metric"), *(len(r[0]) for r in rows))
    w1 = max(len("Train"),  *(len(r[1]) for r in rows))
    w2 = max(len("Val"),    *(len(r[2]) for r in rows))
    w3 = max(len("Holdout"),*(len(r[3]) for r in rows))

    def sep(w0, w1, w2, w3):
        return "+" + "-"*(w0+2) + "+" + "-"*(w1+2) + "+" + "-"*(w2+2) + "+" + "-"*(w3+2) + "+"

    header = sep(w0, w1, w2, w3) + "\n"
    header += f"| {'Metric':<{w0}} | {'Train':>{w1}} | {'Val':>{w2}} | {'Holdout':>{w3}} |\n"
    header += sep(w0, w1, w2, w3)

    body_lines = []
    for label, t, v, h in rows:
        body_lines.append(f"| {label:<{w0}} | {t:>{w1}} | {v:>{w2}} | {h:>{w3}} |")
    body = "\n".join(body_lines) + "\n" + sep(w0, w1, w2, w3)

    # Summary table
    sum_rows = [
        ("Score", fmt_num(current_score)),
        ("Gap",   fmt_num(score_gap)),
        ("LR",    fmt_sci(lr)),
    ]
    sw0 = max(len("Summary"), *(len(r[0]) for r in sum_rows))
    sw1 = max(len("Value"),   *(len(r[1]) for r in sum_rows))

    def ssep(sw0, sw1):
        return "+" + "-"*(sw0+2) + "+" + "-"*(sw1+2) + "+"

    summary = ssep(sw0, sw1) + "\n"
    summary += f"| {'Summary':<{sw0}} | {'Value':>{sw1}} |\n"
    summary += ssep(sw0, sw1) + "\n"
    for k, v in sum_rows:
        summary += f"| {k:<{sw0}} | {v:>{sw1}} |\n"
    summary += ssep(sw0, sw1)

    return f"Epoch << {epoch} >>\n{header}\n{body}\n{summary}"


def log_species_table(logger, train_df, val_df, hold_df, species_col='Species', title='Species Overview'):
    """Log a nicely formatted table of species counts across Train/Val/Hold datasets.

    Shows per-species counts and totals in aligned ASCII table.
    """
    # Safely handle missing column
    for df in (train_df, val_df, hold_df):
        if species_col not in df.columns:
            logger.info(f"{title}: column '{species_col}' not found in one of the dataframes")
            return

    sp_train = train_df[species_col].value_counts()
    sp_val = val_df[species_col].value_counts()
    sp_hold = hold_df[species_col].value_counts()
    all_species = sorted(list(set(sp_train.index) | set(sp_val.index) | set(sp_hold.index)))

    table_header = f"{'Species':<35} | {'Train':>6} | {'Val':>6} | {'Hold':>6} | {'Total':>6}"
    sep = "    " + "-" * len(table_header)

    lines = [f"\n    {'-'*69}", f"    {title}", f"    {'-'*69}", f"    {table_header}", f"{sep}"]

    total_train = total_val = total_hold = 0
    for sp in all_species:
        t = int(sp_train.get(sp, 0))
        v = int(sp_val.get(sp, 0))
        h = int(sp_hold.get(sp, 0))
        tot = t + v + h
        lines.append(f"    {str(sp)[:35]:<35} | {t:>6} | {v:>6} | {h:>6} | {tot:>6}")
        total_train += t
        total_val += v
        total_hold += h

    lines.append(sep)
    lines.append(f"    {'TOTAL':<35} | {total_train:>6} | {total_val:>6} | {total_hold:>6} | {total_train+total_val+total_hold:>6}")
    lines.append(f"    {'-'*69}\n")

    msg = "\n".join(lines)
    logger.info(msg)


def _safe_mean(lst):
    try:
        arr = np.array([float(x) for x in lst if x is not None])
        if arr.size == 0:
            return None
        return float(np.mean(arr))
    except Exception:
        return None


def log_fold_summary_tables(logger, fold, history, best_epoch):
    """Log three tables for a fold:
    - Best-epoch metrics (values at best_epoch)
    - Per-fold epoch averages
    - (Does not compute cross-fold aggregates)"""
    # Keys we care about
    metrics_map = [
        ('Loss (Total)', 'train_loss', 'val_loss', 'holdout_loss'),
        ('Biomass',      'train_bio',  'val_bio',  'holdout_bio'),
        ('Aux',          'train_aux',  'val_aux',  'holdout_aux'),
        ('Species',      'train_sp',   'val_sp',   'holdout_sp'),
        ('HSV',          'train_loss_hsv', 'val_loss_hsv', 'holdout_loss_hsv'),
        ('R2',           'train_r2',   'val_r2',   'holdout_r2'),
    ]

    def get_at(key, idx):
        try:
            if key in history and len(history[key]) > idx and idx >= 0:
                return history[key][idx]
        except Exception:
            pass
        return None

    # Best epoch table
    be_lines = []
    be_lines.append(f"\n    {'='*60}")
    be_lines.append(f"    FOLD {fold} - BEST EPOCH METRICS (epoch={best_epoch})")
    be_lines.append(f"    {'='*60}")
    be_lines.append(f"    {'Metric':<30} | {'Train':>10} | {'Val':>10} | {'Holdout':>10}")
    be_lines.append(f"    " + '-'*64)
    for label, tkey, vkey, hkey in metrics_map:
        t = get_at(tkey, best_epoch)
        v = get_at(vkey, best_epoch)
        h = get_at(hkey, best_epoch)
        t_s = f"{float(t):.4f}" if t is not None else "-"
        v_s = f"{float(v):.4f}" if v is not None else "-"
        h_s = f"{float(h):.4f}" if h is not None else "-"
        be_lines.append(f"    {label:<30} | {t_s:>10} | {v_s:>10} | {h_s:>10}")
    be_lines.append(f"    {'='*60}\n")

    # Per-fold epoch averages
    avg_lines = []
    avg_lines.append(f"    {'='*60}")
    avg_lines.append(f"    FOLD {fold} - EPOCH AVERAGES (mean over epochs)")
    avg_lines.append(f"    {'='*60}")
    avg_lines.append(f"    {'Metric':<30} | {'Train':>10} | {'Val':>10} | {'Holdout':>10}")
    avg_lines.append(f"    " + '-'*64)
    for label, tkey, vkey, hkey in metrics_map:
        t = _safe_mean(history.get(tkey, []))
        v = _safe_mean(history.get(vkey, []))
        h = _safe_mean(history.get(hkey, []))
        t_s = f"{t:.4f}" if t is not None else "-"
        v_s = f"{v:.4f}" if v is not None else "-"
        h_s = f"{h:.4f}" if h is not None else "-"
        avg_lines.append(f"    {label:<30} | {t_s:>10} | {v_s:>10} | {h_s:>10}")
    avg_lines.append(f"    {'='*60}\n")

    # Best-epoch summary (score / gap / lr)
    score = None
    gap = None
    lr = None
    try:
        if 'score' in history and best_epoch >= 0 and best_epoch < len(history['score']):
            score = history['score'][best_epoch]
    except Exception:
        score = None
    try:
        if 'lr' in history:
            lr = _safe_mean(history['lr'])
    except Exception:
        lr = None

    sum_lines = []
    sum_lines.append(f"    {'-'*40}")
    sum_lines.append(f"    Score (best epoch): {score:.4f}" if score is not None else "    Score (best epoch): -")
    sum_lines.append(f"    Avg LR: {lr:.4e}" if lr is not None else "    Avg LR: -")
    sum_lines.append(f"    {'-'*40}\n")

    logger.info("\n" + "\n".join(be_lines + avg_lines + sum_lines))


def log_aggregate_best_across_folds(logger, per_fold_best_list):
    """Given a list of per-fold-best dicts, compute mean/std/min/max and log a compact table.

    Expected dict keys (recommended): 'best_score', 'best_val_loss', 'best_holdout_loss', 'best_val_r2', 'best_holdout_r2'
    """
    if not per_fold_best_list:
        logger.info("No per-fold best metrics to aggregate.")
        return

    # Collect keys
    keys = sorted(set().union(*[set(d.keys()) for d in per_fold_best_list]))
    agg = {}
    for k in keys:
        vals = [d.get(k) for d in per_fold_best_list if d.get(k) is not None]
        try:
            arr = np.array([float(x) for x in vals])
            agg[k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=0)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'n': int(len(arr))
            }
        except Exception:
            agg[k] = None

    # Render table
    lines = []
    lines.append(f"\n    {'='*70}")
    lines.append(f"    AGGREGATED BEST METRICS ACROSS FOLDS (n={len(per_fold_best_list)})")
    lines.append(f"    {'='*70}")
    lines.append(f"    {'Metric':<30} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'N':>3}")
    lines.append(f"    " + '-'*80)
    for k in sorted(agg.keys()):
        v = agg[k]
        if v is None:
            lines.append(f"    {k:<30} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>3}")
        else:
            lines.append(f"    {k:<30} | {v['mean']:10.4f} | {v['std']:10.4f} | {v['min']:10.4f} | {v['max']:10.4f} | {v['n']:3d}")
    lines.append(f"    {'='*70}\n")
    logger.info("\n" + "\n".join(lines))
