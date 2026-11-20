# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""
SAE Batch Experiment Visualization
Focus: backbone -> DiT changes, and shared vs. unique latent spaces
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Global plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Color palette
COLORS = {
    'backbone': '#3498db',  # blue
    'dit': '#e74c3c',       # red
    'shared': '#2ecc71',    # green
    'unique': '#f39c12',    # orange
    'cross': '#9b59b6',     # purple
    'grid': '#ecf0f1',      # light gray
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load experiment results."""
    df = pd.read_csv(csv_path)
    # Create a compact experiment label
    df['exp_label'] = df.apply(
        lambda row: f"raw={'T' if row['use_raw_mse'] else 'F'}_cw={row['cross_weight']}",
        axis=1
    )
    # Add is_random column if not present (for backward compatibility)
    if 'is_random' not in df.columns:
        df['is_random'] = False
    return df


def load_and_merge_results(paired_csv: str, random_csv: str = None) -> pd.DataFrame:
    """
    Load and merge paired (normal) and random baseline results.

    Args:
        paired_csv: Path to the normal paired experiment results
        random_csv: Path to the random baseline results (optional)

    Returns:
        Merged DataFrame with 'is_random' column
    """
    df_paired = load_results(paired_csv)
    if 'is_random' not in df_paired.columns:
        df_paired['is_random'] = False

    if random_csv_paths:
        df_random = pd.concat([load_results(path) for path in random_csv_paths], ignore_index=True)
        if 'is_random' not in df_random.columns:
            df_random['is_random'] = True
        df = pd.concat([df_paired, df_random], ignore_index=True)
        return df

    return df_paired


def plot_comprehensive_comparison(df: pd.DataFrame, output_path: str):
    """
    Comprehensive comparison: backbone vs DiT across key metrics
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Split backbone vs DiT
    df_backbone = df[df['feature'] == 'backbone'].copy()
    df_dit = df[df['feature'] == 'dit'].copy()

    # Sort by cross_weight for cleaner plots
    df_backbone = df_backbone.sort_values('cross_weight')
    df_dit = df_dit.sort_values('cross_weight')

    cross_weights = sorted(df['cross_weight'].unique())
    x_pos = np.arange(len(cross_weights))
    bar_width = 0.35

    # ========== 1. Full reconstruction R² ==========
    ax1 = fig.add_subplot(gs[0, 0])

    for i, use_raw in enumerate([False, True]):
        df_b = df_backbone[df_backbone['use_raw_mse'] == use_raw]
        df_d = df_dit[df_dit['use_raw_mse'] == use_raw]

        offset = (i - 0.5) * 0.2
        marker = 'o' if not use_raw else 's'
        linestyle = '-' if not use_raw else '--'

        ax1.plot(cross_weights, df_b['r2_full_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['backbone'], alpha=0.7 if use_raw else 1.0,
                label=f'Backbone X (raw={use_raw})', linewidth=2, markersize=8)
        ax1.plot(cross_weights, df_d['r2_full_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['dit'], alpha=0.7 if use_raw else 1.0,
                label=f'DiT X (raw={use_raw})', linewidth=2, markersize=8)

    ax1.set_xlabel('Cross Recon Weight', fontsize=11)
    ax1.set_ylabel('R² (Full Reconstruction)', fontsize=11)
    ax1.set_title('Full Reconstruction Quality (X)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # ========== 2. Shared-only R² ==========
    ax2 = fig.add_subplot(gs[0, 1])

    for i, use_raw in enumerate([False, True]):
        df_b = df_backbone[df_backbone['use_raw_mse'] == use_raw]
        df_d = df_dit[df_dit['use_raw_mse'] == use_raw]

        marker = 'o' if not use_raw else 's'
        linestyle = '-' if not use_raw else '--'

        ax2.plot(cross_weights, df_b['r2_shared_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['backbone'], alpha=0.7 if use_raw else 1.0,
                label=f'Backbone (raw={use_raw})', linewidth=2, markersize=8)
        ax2.plot(cross_weights, df_d['r2_shared_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['dit'], alpha=0.7 if use_raw else 1.0,
                label=f'DiT (raw={use_raw})', linewidth=2, markersize=8)

    ax2.set_xlabel('Cross Recon Weight', fontsize=11)
    ax2.set_ylabel('R² (Shared Only)', fontsize=11)
    ax2.set_title('Shared-only Reconstruction Quality (X)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.3, 1.0])

    # ========== 3. Cross-modal gap ==========
    ax3 = fig.add_subplot(gs[0, 2])

    for i, use_raw in enumerate([False, True]):
        df_b = df_backbone[df_backbone['use_raw_mse'] == use_raw]
        df_d = df_dit[df_dit['use_raw_mse'] == use_raw]

        marker = 'o' if not use_raw else 's'
        linestyle = '-' if not use_raw else '--'

        ax3.plot(cross_weights, df_b['cross_diff_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['backbone'], alpha=0.7 if use_raw else 1.0,
                label=f'Backbone (raw={use_raw})', linewidth=2, markersize=8)
        ax3.plot(cross_weights, df_d['cross_diff_x'].values, marker=marker, linestyle=linestyle,
                color=COLORS['dit'], alpha=0.7 if use_raw else 1.0,
                label=f'DiT (raw={use_raw})', linewidth=2, markersize=8)

    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1.5)
    ax3.set_xlabel('Cross Recon Weight', fontsize=11)
    ax3.set_ylabel('Cross Diff (self - cross)', fontsize=11)
    ax3.set_title('Cross-modal Gap (closer to 0 is better)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ========== 4. Shared CKA ==========
    ax4 = fig.add_subplot(gs[1, 0])

    for i, use_raw in enumerate([False, True]):
        df_b = df_backbone[df_backbone['use_raw_mse'] == use_raw]
        df_d = df_dit[df_dit['use_raw_mse'] == use_raw]

        marker = 'o' if not use_raw else 's'
        linestyle = '-' if not use_raw else '--'

        ax4.plot(cross_weights, df_b['cka_shared'].values, marker=marker, linestyle=linestyle,
                color=COLORS['backbone'], alpha=0.7 if use_raw else 1.0,
                label=f'Backbone (raw={use_raw})', linewidth=2, markersize=8)
        ax4.plot(cross_weights, df_d['cka_shared'].values, marker=marker, linestyle=linestyle,
                color=COLORS['dit'], alpha=0.7 if use_raw else 1.0,
                label=f'DiT (raw={use_raw})', linewidth=2, markersize=8)

    ax4.set_xlabel('Cross Recon Weight', fontsize=11)
    ax4.set_ylabel('CKA', fontsize=11)
    ax4.set_title('Shared Space CKA Similarity', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.8, 1.0])

    # ========== 5. Cosine similarity ==========
    ax5 = fig.add_subplot(gs[1, 1])

    for i, use_raw in enumerate([False, True]):
        df_b = df_backbone[df_backbone['use_raw_mse'] == use_raw]
        df_d = df_dit[df_dit['use_raw_mse'] == use_raw]

        marker = 'o' if not use_raw else 's'
        linestyle = '-' if not use_raw else '--'

        ax5.plot(cross_weights, df_b['shared_cosine_sim'].values, marker=marker, linestyle=linestyle,
                color=COLORS['backbone'], alpha=0.7 if use_raw else 1.0,
                label=f'Backbone (raw={use_raw})', linewidth=2, markersize=8)
        ax5.plot(cross_weights, df_d['shared_cosine_sim'].values, marker=marker, linestyle=linestyle,
                color=COLORS['dit'], alpha=0.7 if use_raw else 1.0,
                label=f'DiT (raw={use_raw})', linewidth=2, markersize=8)

    ax5.set_xlabel('Cross Recon Weight', fontsize=11)
    ax5.set_ylabel('Cosine Similarity', fontsize=11)
    ax5.set_title('Shared Space Cosine Similarity', fontsize=13, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0.8, 1.0])

    # ========== 6. Original CKA vs Shared CKA ==========
    ax6 = fig.add_subplot(gs[1, 2])

    # Original CKA (reference lines)
    cka_orig_backbone = df_backbone['cka_original'].iloc[0]
    cka_orig_dit = df_dit['cka_original'].iloc[0]
    ax6.axhline(y=cka_orig_backbone, color=COLORS['backbone'], linestyle=':',
                linewidth=2, label=f'Backbone Original CKA={cka_orig_backbone:.3f}')
    ax6.axhline(y=cka_orig_dit, color=COLORS['dit'], linestyle=':',
                linewidth=2, label=f'DiT Original CKA={cka_orig_dit:.3f}')

    # Shared CKA
    df_b_noraw = df_backbone[df_backbone['use_raw_mse'] == False]
    df_d_noraw = df_dit[df_dit['use_raw_mse'] == False]

    ax6.plot(cross_weights, df_b_noraw['cka_shared'].values, 'o-',
            color=COLORS['backbone'], linewidth=2, markersize=8, label='Backbone Shared CKA')
    ax6.plot(cross_weights, df_d_noraw['cka_shared'].values, 'o-',
            color=COLORS['dit'], linewidth=2, markersize=8, label='DiT Shared CKA')

    ax6.set_xlabel('Cross Recon Weight', fontsize=11)
    ax6.set_ylabel('CKA', fontsize=11)
    ax6.set_title('Original CKA vs Shared-space CKA', fontsize=13, fontweight='bold')
    ax6.legend(loc='center right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.0])

    # ========== 7. Output-space variance decomposition (if available) ==========
    ax7 = fig.add_subplot(gs[2, 0])

    if 'var_x_shared_pct' in df.columns:
        df_b_noraw = df_backbone[df_backbone['use_raw_mse'] == False]
        df_d_noraw = df_dit[df_dit['use_raw_mse'] == False]

        ax7.plot(cross_weights, df_b_noraw['var_x_shared_pct'].values, 'o-',
                color=COLORS['backbone'], linewidth=2, markersize=8, label='Backbone Shared')
        ax7.plot(cross_weights, df_b_noraw['var_x_unique_pct'].values, 's--',
                color=COLORS['backbone'], alpha=0.6, linewidth=2, markersize=8, label='Backbone Unique')
        ax7.plot(cross_weights, df_d_noraw['var_x_shared_pct'].values, 'o-',
                color=COLORS['dit'], linewidth=2, markersize=8, label='DiT Shared')
        ax7.plot(cross_weights, df_d_noraw['var_x_unique_pct'].values, 's--',
                color=COLORS['dit'], alpha=0.6, linewidth=2, markersize=8, label='DiT Unique')

        ax7.set_xlabel('Cross Recon Weight', fontsize=11)
        ax7.set_ylabel('Variance Contribution (%)', fontsize=11)
        ax7.set_title('Output-space Variance Contribution (X)', fontsize=13, fontweight='bold')
        ax7.legend(loc='center right', fontsize=8)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No variance data available', ha='center', va='center',
                transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Output-space Variance Contribution (X)', fontsize=13, fontweight='bold')

    # ========== 8. Backbone vs DiT bar comparison ==========
    ax8 = fig.add_subplot(gs[2, 1])

    # Compare a representative configuration (cross_weight=0.5, raw=False)
    best_backbone = df_backbone[(df_backbone['cross_weight'] == 0.5) & (df_backbone['use_raw_mse'] == False)]
    best_dit = df_dit[(df_dit['cross_weight'] == 0.5) & (df_dit['use_raw_mse'] == False)]

    if len(best_backbone) > 0 and len(best_dit) > 0:
        metrics = ['r2_full_x', 'r2_shared_x', 'r2_x_from_shared_y', 'cka_shared']
        metric_labels = ['Full R²', 'Shared R²', 'Cross R²', 'Shared CKA']

        x_pos = np.arange(len(metrics))

        backbone_vals = [best_backbone[m].values[0] for m in metrics]
        dit_vals = [best_dit[m].values[0] for m in metrics]

        bars1 = ax8.bar(x_pos - 0.2, backbone_vals, 0.35, label='Backbone', color=COLORS['backbone'], alpha=0.8)
        bars2 = ax8.bar(x_pos + 0.2, dit_vals, 0.35, label='DiT', color=COLORS['dit'], alpha=0.8)

        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(metric_labels, fontsize=10)
        ax8.set_ylabel('Value', fontsize=11)
        ax8.set_title('Backbone vs DiT: Key Metrics\n(cross_weight=0.5, raw=False)', fontsize=13, fontweight='bold')
        ax8.legend(loc='lower right', fontsize=10)
        ax8.set_ylim([0, 1.1])
        ax8.grid(True, alpha=0.3, axis='y')

        # Value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax8.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    # ========== 9. Heatmap: cross-modal gap ==========
    ax9 = fig.add_subplot(gs[2, 2])

    # Prepare heatmap data
    heatmap_data = []
    for feature in ['backbone', 'dit']:
        df_f = df[df['feature'] == feature]
        for use_raw in [False, True]:
            df_r = df_f[df_f['use_raw_mse'] == use_raw].sort_values('cross_weight')
            row_label = f"{feature}_raw={'T' if use_raw else 'F'}"
            heatmap_data.append({
                'label': row_label,
                **{f'cw={cw}': df_r[df_r['cross_weight'] == cw]['cross_diff_x'].values[0]
                   for cw in cross_weights}
            })

    heatmap_df = pd.DataFrame(heatmap_data).set_index('label')

    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0,
                ax=ax9, cbar_kws={'label': 'Cross Diff'}, annot_kws={'fontsize': 9})
    ax9.set_title('Cross-modal Gap Heatmap (X)\n(closer to 0 / greener is better)', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Cross Recon Weight', fontsize=11)
    ax9.set_ylabel('Configuration', fontsize=11)

    # Suptitle
    fig.suptitle('SAE Shared/Unique Space Analysis: Backbone vs DiT',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comprehensive comparison figure: {output_path}")


def plot_summary_table_visual(df: pd.DataFrame, output_path: str):
    """
    Render a clean summary table as a figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    # ========== Backbone table ==========
    df_backbone = df[df['feature'] == 'backbone'].copy()
    df_backbone = df_backbone.sort_values(['use_raw_mse', 'cross_weight'])

    # Select key columns (with new cov and resid columns if available)
    base_cols = ['use_raw_mse', 'cross_weight',
                 'r2_full_x', 'r2_shared_x', 'r2_x_from_shared_y', 'cross_diff_x',
                 'cka_shared', 'shared_cosine_sim']
    base_labels = ['Raw MSE', 'Cross W',
                   'Full R²', 'Shared R²', 'Cross R²', 'Cross Diff',
                   'CKA', 'Cosine']

    # Add cov and resid if available
    if 'var_x_cov_pct' in df.columns:
        cols_to_show = base_cols + ['var_x_cov_pct', 'var_x_resid_pct']
        col_labels = base_labels + ['Cov%', 'Resid%']
    else:
        cols_to_show = base_cols
        col_labels = base_labels

    table_data_backbone = df_backbone[cols_to_show].values

    # Format values
    formatted_data = []
    for row in table_data_backbone:
        formatted_row = []
        for i, val in enumerate(row):
            if i == 0:  # use_raw_mse
                formatted_row.append('True' if val else 'False')
            elif i == 1:  # cross_weight
                formatted_row.append(f'{val:.1f}')
            else:
                formatted_row.append(f'{val:.4f}')
        formatted_data.append(formatted_row)

    ax1 = axes[0]
    ax1.axis('off')

    # Create table
    table1 = ax1.table(
        cellText=formatted_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.8)

    # Header style
    for i in range(len(col_labels)):
        table1[(0, i)].set_facecolor(COLORS['backbone'])
        table1[(0, i)].set_text_props(color='white', fontweight='bold')

    # Conditional formatting
    cross_diff_idx = col_labels.index('Cross Diff') if 'Cross Diff' in col_labels else -1
    cov_idx = col_labels.index('Cov%') if 'Cov%' in col_labels else -1
    resid_idx = col_labels.index('Resid%') if 'Resid%' in col_labels else -1

    for i, row in enumerate(formatted_data):
        for j, val in enumerate(row):
            if j >= 2:  # numeric columns
                try:
                    num_val = float(val)
                    if j == cross_diff_idx:  # cross_diff - closer to 0 is better
                        if abs(num_val) < 0.02:
                            table1[(i+1, j)].set_facecolor('#d5f5e3')  # light green
                        elif abs(num_val) < 0.05:
                            table1[(i+1, j)].set_facecolor('#fcf3cf')  # light yellow
                        else:
                            table1[(i+1, j)].set_facecolor('#fadbd8')  # light red
                    elif j == cov_idx or j == resid_idx:  # cov/resid - closer to 0 is better
                        if abs(num_val) < 5:
                            table1[(i+1, j)].set_facecolor('#d5f5e3')
                        elif abs(num_val) < 15:
                            table1[(i+1, j)].set_facecolor('#fcf3cf')
                        else:
                            table1[(i+1, j)].set_facecolor('#fadbd8')
                    else:  # other metrics - higher is better
                        if num_val > 0.9:
                            table1[(i+1, j)].set_facecolor('#d5f5e3')
                        elif num_val > 0.7:
                            table1[(i+1, j)].set_facecolor('#fcf3cf')
                        elif num_val > 0.5:
                            table1[(i+1, j)].set_facecolor('#fdebd0')
                except:
                    pass

    ax1.set_title('Backbone Feature Results', fontsize=14, fontweight='bold',
                  color=COLORS['backbone'], pad=20)

    # ========== DiT table ==========
    df_dit = df[df['feature'] == 'dit'].copy()
    df_dit = df_dit.sort_values(['use_raw_mse', 'cross_weight'])

    table_data_dit = df_dit[cols_to_show].values

    formatted_data_dit = []
    for row in table_data_dit:
        formatted_row = []
        for i, val in enumerate(row):
            if i == 0:
                formatted_row.append('True' if val else 'False')
            elif i == 1:
                formatted_row.append(f'{val:.1f}')
            else:
                formatted_row.append(f'{val:.4f}')
        formatted_data_dit.append(formatted_row)

    ax2 = axes[1]
    ax2.axis('off')

    table2 = ax2.table(
        cellText=formatted_data_dit,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.8)

    for i in range(len(col_labels)):
        table2[(0, i)].set_facecolor(COLORS['dit'])
        table2[(0, i)].set_text_props(color='white', fontweight='bold')

    for i, row in enumerate(formatted_data_dit):
        for j, val in enumerate(row):
            if j >= 2:
                try:
                    num_val = float(val)
                    if j == cross_diff_idx:
                        if abs(num_val) < 0.02:
                            table2[(i+1, j)].set_facecolor('#d5f5e3')
                        elif abs(num_val) < 0.05:
                            table2[(i+1, j)].set_facecolor('#fcf3cf')
                        else:
                            table2[(i+1, j)].set_facecolor('#fadbd8')
                    elif j == cov_idx or j == resid_idx:
                        if abs(num_val) < 5:
                            table2[(i+1, j)].set_facecolor('#d5f5e3')
                        elif abs(num_val) < 15:
                            table2[(i+1, j)].set_facecolor('#fcf3cf')
                        else:
                            table2[(i+1, j)].set_facecolor('#fadbd8')
                    else:
                        if num_val > 0.9:
                            table2[(i+1, j)].set_facecolor('#d5f5e3')
                        elif num_val > 0.7:
                            table2[(i+1, j)].set_facecolor('#fcf3cf')
                        elif num_val > 0.5:
                            table2[(i+1, j)].set_facecolor('#fdebd0')
                except:
                    pass

    ax2.set_title('DiT Feature Results', fontsize=14, fontweight='bold',
                  color=COLORS['dit'], pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d5f5e3', edgecolor='gray', label='Excellent (>0.9 or |diff|<0.02)'),
        mpatches.Patch(facecolor='#fcf3cf', edgecolor='gray', label='Good (>0.7 or |diff|<0.05)'),
        mpatches.Patch(facecolor='#fadbd8', edgecolor='gray', label='Needs improvement'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('SAE Batch Experiment Summary Table', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved summary table figure: {output_path}")


def plot_backbone_vs_dit_radar(df: pd.DataFrame, output_path: str):
    """
    Radar chart: Backbone vs DiT multi-metric comparison
    """
    from math import pi

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

    # Pick a representative config (cross_weight=0.5, raw=False)
    df_best = df[(df['cross_weight'] == 0.5) & (df['use_raw_mse'] == False)]

    if len(df_best) < 2:
        # Fallback to the smallest cross_weight
        df_best = df[df['cross_weight'] == df['cross_weight'].min()]

    backbone_data = df_best[df_best['feature'] == 'backbone'].iloc[0]
    dit_data = df_best[df_best['feature'] == 'dit'].iloc[0]

    # Radar dimensions
    categories = ['Full R² (X)', 'Shared R² (X)', 'Cross R² (X)',
                  'Shared CKA', 'Cosine Sim']

    backbone_values = [
        backbone_data['r2_full_x'],
        backbone_data['r2_shared_x'],
        backbone_data['r2_x_from_shared_y'],
        backbone_data['cka_shared'],
        backbone_data['shared_cosine_sim'],
    ]

    dit_values = [
        dit_data['r2_full_x'],
        dit_data['r2_shared_x'],
        dit_data['r2_x_from_shared_y'],
        dit_data['cka_shared'],
        dit_data['shared_cosine_sim'],
    ]

    # Radar angles
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the loop

    backbone_values += backbone_values[:1]
    dit_values += dit_values[:1]

    # Backbone
    ax1 = axes[0]
    ax1.plot(angles, backbone_values, 'o-', linewidth=2, color=COLORS['backbone'], label='Backbone')
    ax1.fill(angles, backbone_values, alpha=0.25, color=COLORS['backbone'])
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylim([0, 1])
    ax1.set_title('Backbone Features', fontsize=13, fontweight='bold', color=COLORS['backbone'])
    ax1.grid(True)

    # DiT
    ax2 = axes[1]
    ax2.plot(angles, dit_values, 'o-', linewidth=2, color=COLORS['dit'], label='DiT')
    ax2.fill(angles, dit_values, alpha=0.25, color=COLORS['dit'])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.set_title('DiT Features', fontsize=13, fontweight='bold', color=COLORS['dit'])
    ax2.grid(True)

    fig.suptitle('Backbone vs DiT Multi-metric Comparison (cross_weight=0.5)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved radar chart figure: {output_path}")


def plot_cross_weight_effect(df: pd.DataFrame, output_path: str):
    """
    Cross weight effect analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, feature in enumerate(['backbone', 'dit']):
        df_f = df[(df['feature'] == feature) & (df['use_raw_mse'] == False)]
        df_f = df_f.sort_values('cross_weight')

        color = COLORS['backbone'] if feature == 'backbone' else COLORS['dit']

        # Row 1: reconstruction quality
        ax = axes[0, idx]
        ax.plot(df_f['cross_weight'], df_f['r2_full_x'], 'o-',
                color=color, linewidth=2, markersize=10, label='Full R² (X)')
        ax.plot(df_f['cross_weight'], df_f['r2_shared_x'], 's--',
                color=COLORS['shared'], linewidth=2, markersize=10, label='Shared R² (X)')
        ax.plot(df_f['cross_weight'], df_f['r2_x_from_shared_y'], '^:',
                color=COLORS['cross'], linewidth=2, markersize=10, label='Cross R² (X)')

        ax.set_xlabel('Cross Recon Weight', fontsize=11)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title(f'{feature.upper()} - Reconstruction vs Cross Weight',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])

        # Gap shading
        ax.fill_between(df_f['cross_weight'],
                        df_f['r2_shared_x'], df_f['r2_x_from_shared_y'],
                        alpha=0.2, color='gray', label='Self-Cross Gap')

        # Row 2: cross-modal gap
        ax = axes[1, idx]
        ax.bar(df_f['cross_weight'], df_f['cross_diff_x'],
               width=0.08, color=color, alpha=0.7, label='X')
        ax.bar(df_f['cross_weight'] + 0.1, df_f['cross_diff_y'],
               width=0.08, color=color, alpha=0.4, label='Y')

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Cross Recon Weight', fontsize=11)
        ax.set_ylabel('Cross Diff (self - cross)', fontsize=11)
        ax.set_title(f'{feature.upper()} - Cross-modal Gap', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Cross Recon Weight Effect (raw_mse=False)',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cross-weight effect figure: {output_path}")


def plot_paired_vs_random_comparison(df: pd.DataFrame, output_path: str):
    """
    Compare paired (normal) vs random baseline experiments.
    Shows how key metrics differ when sample pairing is shuffled.
    """
    # Check if we have random baseline data
    if 'is_random' not in df.columns or df['is_random'].nunique() < 2:
        print(f"Skipping paired vs random comparison: no random baseline data")
        return

    df_paired = df[df['is_random'] == False].copy()
    df_random = df[df['is_random'] == True].copy()

    if len(df_paired) == 0 or len(df_random) == 0:
        print(f"Skipping paired vs random comparison: missing paired or random data")
        return

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Metrics to compare
    metrics = [
        ('cka_shared', 'Shared CKA', 'Higher is better'),
        ('shared_cosine_sim', 'Shared Cosine Similarity', 'Higher is better'),
        ('r2_x_from_shared_y', 'Cross R² (X from shared_Y)', 'Higher is better'),
        ('r2_y_from_shared_x', 'Cross R² (Y from shared_X)', 'Higher is better'),
        ('cross_diff_x', 'Cross Diff X (self - cross)', 'Lower is better'),
        ('cross_diff_y', 'Cross Diff Y (self - cross)', 'Lower is better'),
    ]

    for i, (metric, title, note) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 3, i % 3])

        for feat_idx, feature in enumerate(['backbone', 'dit']):
            df_p = df_paired[df_paired['feature'] == feature].sort_values('cross_weight')
            df_r = df_random[df_random['feature'] == feature].sort_values('cross_weight')

            # Only use raw_mse=False for cleaner comparison
            df_p = df_p[df_p['use_raw_mse'] == False]
            df_r = df_r[df_r['use_raw_mse'] == False]

            if len(df_p) == 0 or len(df_r) == 0:
                continue

            color = COLORS['backbone'] if feature == 'backbone' else COLORS['dit']

            # Paired (solid line)
            ax.plot(df_p['cross_weight'], df_p[metric], 'o-',
                    color=color, linewidth=2.5, markersize=10,
                    label=f'{feature.upper()} (Paired)')

            # Random (dashed line)
            ax.plot(df_r['cross_weight'], df_r[metric], 's--',
                    color=color, linewidth=2, markersize=8, alpha=0.6,
                    label=f'{feature.upper()} (Random)')

        ax.set_xlabel('Cross Recon Weight', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{title}\n({note})', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Paired vs Random Baseline Comparison\n(Validating meaningful cross-modal correspondence)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved paired vs random comparison: {output_path}")


def plot_random_baseline_summary(df: pd.DataFrame, output_path: str):
    """
    Summary bar chart comparing paired vs random for key metrics.
    Shows the "gap" between paired and random as validation of learned correspondence.
    """
    if 'is_random' not in df.columns or df['is_random'].nunique() < 2:
        print(f"Skipping random baseline summary: no random baseline data")
        return

    df_paired = df[df['is_random'] == False].copy()
    df_random = df[df['is_random'] == True].copy()

    if len(df_paired) == 0 or len(df_random) == 0:
        print(f"Skipping random baseline summary: missing paired or random data")
        return

    # Use best configuration (cross_weight=0.5, raw_mse=False) for comparison
    df_p = df_paired[(df_paired['cross_weight'] == 0.5) & (df_paired['use_raw_mse'] == False)]
    df_r = df_random[(df_random['cross_weight'] == 0.5) & (df_random['use_raw_mse'] == False)]

    if len(df_p) == 0 or len(df_r) == 0:
        # Fallback: use any available cross_weight
        df_p = df_paired[df_paired['use_raw_mse'] == False].groupby('feature').first().reset_index()
        df_r = df_random[df_random['use_raw_mse'] == False].groupby('feature').first().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_compare = [
        ('cka_shared', 'Shared CKA'),
        ('shared_cosine_sim', 'Shared Cosine Sim'),
        ('r2_x_from_shared_y', 'Cross R² (X←shared_Y)'),
    ]

    for ax, (metric, title) in zip(axes, metrics_to_compare):
        x = np.arange(2)  # backbone, dit
        width = 0.35

        paired_vals = [df_p[df_p['feature'] == f][metric].values[0] if len(df_p[df_p['feature'] == f]) > 0 else 0
                       for f in ['backbone', 'dit']]
        random_vals = [df_r[df_r['feature'] == f][metric].values[0] if len(df_r[df_r['feature'] == f]) > 0 else 0
                       for f in ['backbone', 'dit']]

        bars1 = ax.bar(x - width/2, paired_vals, width, label='Paired', color=[COLORS['backbone'], COLORS['dit']])
        bars2 = ax.bar(x + width/2, random_vals, width, label='Random', color=[COLORS['backbone'], COLORS['dit']], alpha=0.4, hatch='//')

        # Add gap annotations
        for i, (p, r) in enumerate(zip(paired_vals, random_vals)):
            gap = p - r
            gap_pct = (gap / r * 100) if r != 0 else 0
            ax.annotate(f'Δ={gap:.3f}\n({gap_pct:+.1f}%)',
                       xy=(i, max(p, r) + 0.02), ha='center', fontsize=9, fontweight='bold',
                       color='green' if gap > 0 else 'red')

        ax.set_xticks(x)
        ax.set_xticklabels(['Backbone', 'DiT'])
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

    fig.suptitle('Paired vs Random Baseline Summary (cross_weight=0.5, raw_mse=False)\n'
                 'Positive gap validates learned cross-modal correspondence',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved random baseline summary: {output_path}")


def plot_variance_decomposition_detailed(df: pd.DataFrame, output_path: str):
    """
    Detailed variance decomposition plot showing shared, unique, cov, and residual.
    """
    if 'var_x_cov_pct' not in df.columns:
        print(f"Skipping variance decomposition plot: cov/resid columns not found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    cross_weights = sorted(df['cross_weight'].unique())

    for idx, feature in enumerate(['backbone', 'dit']):
        df_f = df[(df['feature'] == feature) & (df['use_raw_mse'] == False)]
        df_f = df_f.sort_values('cross_weight')

        color = COLORS['backbone'] if feature == 'backbone' else COLORS['dit']

        # X variance decomposition
        ax = axes[0, idx]
        width = 0.18
        x_pos = np.arange(len(cross_weights))

        bars_shared = ax.bar(x_pos - 1.5*width, df_f['var_x_shared_pct'].values, width,
                            label='Shared', color=COLORS['shared'], alpha=0.8)
        bars_unique = ax.bar(x_pos - 0.5*width, df_f['var_x_unique_pct'].values, width,
                            label='Unique', color=COLORS['unique'], alpha=0.8)
        bars_cov = ax.bar(x_pos + 0.5*width, df_f['var_x_cov_pct'].values, width,
                         label='Cov (2×)', color=COLORS['cross'], alpha=0.8)
        bars_resid = ax.bar(x_pos + 1.5*width, df_f['var_x_resid_pct'].values, width,
                           label='Residual', color='gray', alpha=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{cw}' for cw in cross_weights])
        ax.set_xlabel('Cross Recon Weight', fontsize=11)
        ax.set_ylabel('Variance Contribution (%)', fontsize=11)
        ax.set_title(f'{feature.upper()} - X Variance Decomposition', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Y variance decomposition
        ax = axes[1, idx]

        bars_shared = ax.bar(x_pos - 1.5*width, df_f['var_y_shared_pct'].values, width,
                            label='Shared', color=COLORS['shared'], alpha=0.8)
        bars_unique = ax.bar(x_pos - 0.5*width, df_f['var_y_unique_pct'].values, width,
                            label='Unique', color=COLORS['unique'], alpha=0.8)
        bars_cov = ax.bar(x_pos + 0.5*width, df_f['var_y_cov_pct'].values, width,
                         label='Cov (2×)', color=COLORS['cross'], alpha=0.8)
        bars_resid = ax.bar(x_pos + 1.5*width, df_f['var_y_resid_pct'].values, width,
                           label='Residual', color='gray', alpha=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{cw}' for cw in cross_weights])
        ax.set_xlabel('Cross Recon Weight', fontsize=11)
        ax.set_ylabel('Variance Contribution (%)', fontsize=11)
        ax.set_title(f'{feature.upper()} - Y Variance Decomposition', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    fig.suptitle('Variance Decomposition: Shared + Unique + Cov + Residual (raw_mse=False)',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved variance decomposition figure: {output_path}")


def generate_all_visualizations(csv_paths, output_dir: str, random_csv_paths=None):
    """
    Generate all visualizations.

    Args:
        csv_paths: Paired experiment results CSV files
        output_dir: Output directory for figures
        random_csv_paths: Optional random baseline results CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Loading experiment results...")
    print(f"{'='*60}")

    df_paired = pd.concat([load_results(path) for path in csv_paths], ignore_index=True)
    df_paired["is_random"] = False
    df = df_paired
    if random_csv_paths:
        df_random = pd.concat([load_results(path) for path in random_csv_paths], ignore_index=True)
        df_random["is_random"] = True
        df = pd.concat([df_paired, df_random], ignore_index=True)

    print(f"Feature types: {df['feature'].unique()}")
    print(f"Cross weights: {sorted(df['cross_weight'].unique())}")

    # Check for special columns
    has_cov_resid = 'var_x_cov_pct' in df.columns
    has_random = 'is_random' in df.columns and df['is_random'].any()
    print(f"Has cov/resid columns: {has_cov_resid}")
    print(f"Has random baseline data: {has_random}")

    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    # 1) Comprehensive comparison
    plot_comprehensive_comparison(
        df_paired,
        os.path.join(output_dir, "sae_comprehensive_comparison.png")
    )

    # 2) Summary table
    plot_summary_table_visual(
        df_paired,
        os.path.join(output_dir, "sae_summary_table.png")
    )

    # 3) Radar chart
    plot_backbone_vs_dit_radar(
        df_paired,
        os.path.join(output_dir, "sae_radar_comparison.png")
    )

    # 4) Cross weight effect
    plot_cross_weight_effect(
        df_paired,
        os.path.join(output_dir, "sae_cross_weight_effect.png")
    )

    # 5) Variance decomposition (if cov/resid columns exist)
    if has_cov_resid:
        plot_variance_decomposition_detailed(
            df_paired,
            os.path.join(output_dir, "sae_variance_decomposition.png")
        )

    # 6) Random baseline comparison (if random data exists)
    if has_random:
        plot_paired_vs_random_comparison(
            df,
            os.path.join(output_dir, "sae_paired_vs_random_comparison.png")
        )

        plot_random_baseline_summary(
            df,
            os.path.join(output_dir, "sae_random_baseline_summary.png")
        )

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    print(f"All figures saved to: {output_dir}/")
    print("Generated files:")
    print("  - sae_comprehensive_comparison.png  (comprehensive comparison)")
    print("  - sae_summary_table.png             (summary table)")
    print("  - sae_radar_comparison.png          (radar chart)")
    print("  - sae_cross_weight_effect.png       (cross weight effect)")
    if has_cov_resid:
        print("  - sae_variance_decomposition.png    (variance decomposition)")
    if has_random:
        print("  - sae_paired_vs_random_comparison.png (paired vs random comparison)")
        print("  - sae_random_baseline_summary.png   (random baseline summary)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SAE experiment result visualization')
    parser.add_argument('--csv', nargs='+',
                        required=True,
                        help='Path to the main (paired) result CSV file')
    parser.add_argument('--random-csv', nargs='+',
                        default=None,
                        help='Path to the random baseline result CSV file (optional)')
    parser.add_argument('--output', type=str,
                        required=True,
                        help='Output directory for figures')

    args = parser.parse_args()

    generate_all_visualizations(args.csv, args.output, random_csv_paths=args.random_csv)
