# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""
生成 Backbone vs DiT CKA 矩阵对比图
- 横向排列两个 CKA 热图
- 无子图标题和大图标题
- 优化字体颜色（0.5 附近使用黑色）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pca_whitening_cca import linear_cka

# 配色
MODEL_DISPLAY_NAMES = {
    "VLM": "VLM",
    "ViT-Large": "ViT-Large",
    "ResNet101": "ResNet-101",
    "EVA-CLIP": "EVA-CLIP",
}


def compute_cka_matrix(features_dict, model_names):
    """计算 CKA 相似度矩阵"""
    n = len(model_names)
    cka_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cka_val = linear_cka(features_dict[model_names[i]], features_dict[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
    return cka_matrix


def get_text_color(value):
    """根据 CKA 值选择字体颜色，0.4-0.6 之间使用黑色"""
    if 0.35 <= value <= 0.65:
        return 'black'
    elif value > 0.65:
        return 'white'
    else:
        return 'black'


def plot_cka_heatmap(ax, cka_matrix, model_names, label=None):
    """绘制单个 CKA 热图"""
    display_names = [MODEL_DISPLAY_NAMES[name] for name in model_names]

    im = ax.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')

    # 添加数值标注
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            value = cka_matrix[i, j]
            color = get_text_color(value)
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    # 设置刻度
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(display_names, fontsize=12, rotation=30, ha='right')
    ax.set_yticklabels(display_names, fontsize=12)

    # 添加左下角标签（Backbone / DiT）
    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    return im


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize archived multi-model representations.")
    parser.add_argument("--cache-dir", required=True, help="Directory of aligned multi-model NPZs.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    output_dir = args.output
    cache_dir = args.cache_dir
    model_names = ["VLM", "ViT-Large", "ResNet101", "EVA-CLIP"]

    print("=" * 60)
    print("生成 Backbone vs DiT CKA 矩阵对比图")
    print("=" * 60)

    # 加载 Backbone 特征
    cache_backbone = os.path.join(cache_dir, "all_models_aligned_cache.npz")
    cache_dit = os.path.join(cache_dir, "all_models_aligned_cache_dit.npz")

    cka_backbone = None
    cka_dit = None

    # 加载或计算 Backbone CKA
    if os.path.exists(cache_backbone):
        print(f"\n加载 Backbone 特征: {cache_backbone}")
        data = np.load(cache_backbone, allow_pickle=True)
        features_backbone = {name: data[name] for name in model_names}
        print(f"  样本数: {features_backbone[model_names[0]].shape[0]}")

        print("  计算 Backbone CKA...")
        cka_backbone = compute_cka_matrix(features_backbone, model_names)
        print("  Backbone CKA 矩阵:")
        for i, name in enumerate(model_names):
            print(f"    {name:<12}" + "".join([f"{cka_backbone[i,j]:>8.4f}" for j in range(len(model_names))]))
    else:
        print(f"\n警告: Backbone 缓存不存在 {cache_backbone}")

    # 加载或计算 DiT CKA
    if os.path.exists(cache_dit):
        print(f"\n加载 DiT 特征: {cache_dit}")
        data = np.load(cache_dit, allow_pickle=True)
        features_dit = {name: data[name] for name in model_names}
        print(f"  样本数: {features_dit[model_names[0]].shape[0]}")

        print("  计算 DiT CKA...")
        cka_dit = compute_cka_matrix(features_dit, model_names)
        print("  DiT CKA 矩阵:")
        for i, name in enumerate(model_names):
            print(f"    {name:<12}" + "".join([f"{cka_dit[i,j]:>8.4f}" for j in range(len(model_names))]))
    else:
        print(f"\n警告: DiT 缓存不存在 {cache_dit}")

    # 检查是否有数据
    if cka_backbone is None and cka_dit is None:
        print("\n错误: 没有找到任何 CKA 数据")
        return

    # 绘图
    print("\n生成对比图...")

    if cka_backbone is not None and cka_dit is not None:
        # 两个都有，横向排列
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        im1 = plot_cka_heatmap(ax1, cka_backbone, model_names, label="Backbone")
        im2 = plot_cka_heatmap(ax2, cka_dit, model_names, label="DiT")

        # 添加共享 colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('CKA Similarity', fontsize=12)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

    elif cka_backbone is not None:
        # 只有 Backbone
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = plot_cka_heatmap(ax, cka_backbone, model_names, label="Backbone")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('CKA Similarity', fontsize=12)
        plt.tight_layout()

    else:
        # 只有 DiT
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = plot_cka_heatmap(ax, cka_dit, model_names, label="DiT")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('CKA Similarity', fontsize=12)
        plt.tight_layout()

    # 保存
    save_path = os.path.join(output_dir, "cka_backbone_vs_dit.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n完成！图片保存到: {save_path}")

    # 打印 CKA 差异（如果两者都存在）
    if cka_backbone is not None and cka_dit is not None:
        print("\n" + "=" * 60)
        print("CKA 变化 (DiT - Backbone):")
        print("=" * 60)
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                diff = cka_dit[i, j] - cka_backbone[i, j]
                arrow = "↑" if diff > 0 else "↓"
                print(f"  {model_names[i]:<12} - {model_names[j]:<12}: "
                      f"{cka_backbone[i,j]:.4f} -> {cka_dit[i,j]:.4f} ({arrow}{abs(diff):.4f})")


if __name__ == "__main__":
    main()
