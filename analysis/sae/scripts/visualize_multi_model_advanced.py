# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""
多模型表征空间高级可视化脚本

更专业、更具解释性的可视化方案：
- 方案 A: Procrustes 对齐 + UMAP + 同场景连线
- 方案 C: 层次聚类 dendrogram
- 方案 D: matched vs random 相似度分布 + 1-NN retrieval accuracy
"""

import os
import numpy as np
import time

# 非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("警告: UMAP 未安装，将使用 PCA 替代")

from analyze_pca_whitening_cca import linear_cka

# 尝试导入 PyTorch 和 SAE
try:
    import torch
    from analyze_sae_shared_unique import SharedUniqueSAE
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 或 SAE 模块未找到，SAE 可视化功能将不可用")

# ============ 配色方案 ============
MODEL_COLORS = {
    "VLM": "#E74C3C",
    "ViT-Large": "#3498DB",
    "ResNet101": "#2ECC71",
    "EVA-CLIP": "#9B59B6",
}

MODEL_DISPLAY_NAMES = {
    "VLM": "VLM",
    "ViT-Large": "ViT-Large",
    "ResNet101": "ResNet-101",
    "EVA-CLIP": "EVA-CLIP",
}


# ============ 方案 A: Procrustes 对齐 ============

def fit_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes: 学习正交变换 W 使得 X @ W ≈ Y

    参数:
        X: 源特征 [n, d]
        Y: 目标特征 [n, d]（参考坐标系）
    返回:
        W: 正交变换矩阵 [d, d]
    """
    # 中心化
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # SVD
    M = X_centered.T @ Y_centered
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt

    return W


def align_to_reference(
    features_dict: Dict[str, np.ndarray],
    model_names: List[str],
    reference_model: str = "ResNet101",
) -> Dict[str, np.ndarray]:
    """
    将所有模型对齐到参考模型的坐标系

    返回: aligned_features_dict
    """
    print(f"\n[Procrustes 对齐] 参考模型: {reference_model}")

    aligned = {}
    ref_feats = features_dict[reference_model]

    # 标准化参考模型
    scaler_ref = StandardScaler()
    ref_scaled = scaler_ref.fit_transform(ref_feats)
    aligned[reference_model] = ref_scaled

    for name in model_names:
        if name == reference_model:
            continue

        # 标准化源模型
        scaler_src = StandardScaler()
        src_scaled = scaler_src.fit_transform(features_dict[name])

        # 学习 Procrustes 变换
        W = fit_procrustes(src_scaled, ref_scaled)

        # 对齐
        aligned_feats = src_scaled @ W
        aligned[name] = aligned_feats

        # 计算对齐后的 MSE
        mse = np.mean((aligned_feats - ref_scaled) ** 2)
        print(f"  {name} -> {reference_model}: MSE = {mse:.4f}")

    return aligned


def plot_procrustes_aligned_umap(
    features_dict: Dict[str, np.ndarray],
    model_names: List[str],
    save_path: str,
    reference_model: str = "ResNet101",
    n_samples: int = 1000,
    show_arrows: bool = True,
    arrow_sample_ratio: float = 0.05,
) -> None:
    """
    方案 A: Procrustes 对齐后的 UMAP 可视化，带同场景连线
    """
    print(f"\n[方案 A] Procrustes + UMAP 可视化")
    start_time = time.time()
    np.random.seed(42)

    # 对齐
    aligned = align_to_reference(features_dict, model_names, reference_model)

    # 采样
    n_total = features_dict[model_names[0]].shape[0]
    n_samples = min(n_samples, n_total)
    sample_idx = np.random.choice(n_total, n_samples, replace=False)

    # 合并对齐后的特征
    all_aligned = np.vstack([aligned[name][sample_idx] for name in model_names])

    # UMAP 或 PCA
    print(f"  运行降维 ({n_samples * len(model_names)} 样本)...")
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42, verbose=True)
        coords = reducer.fit_transform(all_aligned)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_aligned)

    # 分离每个模型的坐标
    model_coords = {}
    start_idx = 0
    for name in model_names:
        end_idx = start_idx + n_samples
        model_coords[name] = coords[start_idx:end_idx]
        start_idx = end_idx

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制连线（同场景跨模型位移）
    if show_arrows:
        n_arrows = int(n_samples * arrow_sample_ratio)
        arrow_idx = np.random.choice(n_samples, n_arrows, replace=False)

        print(f"  绘制 {n_arrows} 条同场景连线...")
        ref_coords = model_coords[reference_model]

        for name in model_names:
            if name == reference_model:
                continue

            other_coords = model_coords[name]
            color = MODEL_COLORS[name]

            for i in arrow_idx:
                ax.plot([ref_coords[i, 0], other_coords[i, 0]],
                       [ref_coords[i, 1], other_coords[i, 1]],
                       color=color, alpha=0.15, linewidth=0.5, zorder=1)

    # 绘制点云
    for name in model_names:
        ax.scatter(model_coords[name][:, 0], model_coords[name][:, 1],
                  c=MODEL_COLORS[name], s=8, alpha=0.6,
                  label=MODEL_DISPLAY_NAMES[name], zorder=2)

    # 绘制每个模型的中心点
    for name in model_names:
        center = model_coords[name].mean(axis=0)
        ax.scatter(center[0], center[1], c=MODEL_COLORS[name],
                  s=300, marker='X', edgecolors='white', linewidths=2, zorder=3)

    method = "UMAP (cosine)" if HAS_UMAP else "PCA"
    ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
    ax.set_title(f'Procrustes-Aligned Feature Space\n(Reference: {reference_model}, Lines = Same Scene Displacement)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[方案 A] 完成! 耗时: {time.time() - start_time:.1f}s")
    print(f"  保存到: {save_path}")


# ============ 方案 C: 层次聚类 Dendrogram ============

def plot_hierarchical_clustering(
    cka_matrix: np.ndarray,
    model_names: List[str],
    save_path: str,
) -> None:
    """
    方案 C: 基于 CKA 距离的层次聚类 dendrogram
    """
    print(f"\n[方案 C] 层次聚类 Dendrogram")

    # CKA -> 距离
    distance_matrix = np.sqrt(2 * (1 - cka_matrix))

    # 转换为 condensed 形式
    condensed_dist = squareform(distance_matrix)

    # 层次聚类
    Z = linkage(condensed_dist, method='average')

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    display_names = [MODEL_DISPLAY_NAMES[name] for name in model_names]

    # 自定义颜色
    def color_func(k):
        return '#2C3E50'

    dendrogram(
        Z,
        labels=display_names,
        ax=ax,
        leaf_font_size=12,
        color_threshold=0,
        above_threshold_color='#2C3E50',
    )

    # 在叶子节点添加颜色标记
    xlocs = ax.get_xticks()
    for i, (xloc, name) in enumerate(zip(xlocs, model_names)):
        ax.scatter(xloc, -0.02, c=MODEL_COLORS[name], s=200, zorder=10, clip_on=False)

    ax.set_ylabel('Distance (√(2(1-CKA)))', fontsize=12)
    ax.set_title('Hierarchical Clustering of Model Representations\n(Based on CKA Similarity)',
                fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加 CKA 值标注
    textstr = 'CKA Values:\n'
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            textstr += f'{model_names[i][:3]}-{model_names[j][:3]}: {cka_matrix[i,j]:.3f}\n'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[方案 C] 完成!")
    print(f"  保存到: {save_path}")


# ============ 方案 D: Matched vs Random 相似度分布 ============

def compute_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """计算逐样本的余弦相似度"""
    # 归一化
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    # 逐行点积
    return np.sum(X_norm * Y_norm, axis=1)


def compute_1nn_retrieval_accuracy(X: np.ndarray, Y: np.ndarray) -> float:
    """
    计算 1-NN 检索准确率
    对 X 中每个样本，在 Y 中找最近邻，看是否命中同索引
    """
    # 归一化
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)

    # 计算相似度矩阵
    sim_matrix = X_norm @ Y_norm.T  # [n, n]

    # 找每行最大值的索引
    nn_indices = np.argmax(sim_matrix, axis=1)

    # 计算准确率
    correct = np.sum(nn_indices == np.arange(len(X)))
    accuracy = correct / len(X)

    return accuracy


def plot_similarity_distribution(
    features_dict: Dict[str, np.ndarray],
    model_names: List[str],
    save_path: str,
    n_samples: int = 5000,
) -> None:
    """
    方案 D1: Matched vs Random 相似度分布
    """
    print(f"\n[方案 D1] Matched vs Random 相似度分布")
    np.random.seed(42)

    n_total = features_dict[model_names[0]].shape[0]
    n_samples = min(n_samples, n_total)
    sample_idx = np.random.choice(n_total, n_samples, replace=False)

    # 生成所有模型对
    pairs = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            pairs.append((model_names[i], model_names[j]))

    # 计算相似度
    results = {}
    for name_a, name_b in pairs:
        X = features_dict[name_a][sample_idx]
        Y = features_dict[name_b][sample_idx]

        # Matched (同场景)
        matched_sim = compute_cosine_similarity(X, Y)

        # Random (打乱)
        perm = np.random.permutation(n_samples)
        random_sim = compute_cosine_similarity(X, Y[perm])

        results[(name_a, name_b)] = {
            'matched': matched_sim,
            'random': random_sim,
        }

    # 绘图
    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name_a, name_b) in enumerate(pairs):
        ax = axes[idx]

        matched = results[(name_a, name_b)]['matched']
        random = results[(name_a, name_b)]['random']

        # 直方图
        ax.hist(random, bins=50, alpha=0.6, color='gray', label='Random (Mismatched)', density=True)
        ax.hist(matched, bins=50, alpha=0.6, color=MODEL_COLORS[name_a], label='Matched (Same Scene)', density=True)

        # 统计信息
        sep = matched.mean() - random.mean()
        ax.axvline(matched.mean(), color=MODEL_COLORS[name_a], linestyle='--', linewidth=2)
        ax.axvline(random.mean(), color='gray', linestyle='--', linewidth=2)

        ax.set_xlabel('Cosine Similarity', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{MODEL_DISPLAY_NAMES[name_a]} vs {MODEL_DISPLAY_NAMES[name_b]}\n'
                    f'Separation: {sep:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Matched vs Random Similarity Distribution\n(Higher Separation = Stronger Cross-Model Alignment)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[方案 D1] 完成!")
    print(f"  保存到: {save_path}")


def plot_retrieval_accuracy(
    features_dict: Dict[str, np.ndarray],
    model_names: List[str],
    save_path: str,
    n_samples: int = 5000,
) -> None:
    """
    方案 D2: 1-NN 检索准确率条形图
    """
    print(f"\n[方案 D2] 1-NN 检索准确率")
    np.random.seed(42)

    n_total = features_dict[model_names[0]].shape[0]
    n_samples = min(n_samples, n_total)
    sample_idx = np.random.choice(n_total, n_samples, replace=False)

    # 生成所有模型对
    pairs = []
    accuracies = []
    colors = []

    # VLM vs 视觉
    vlm_idx = model_names.index("VLM")
    visual_names = [n for n in model_names if n != "VLM"]

    # 先计算 VLM vs Visual
    for name in visual_names:
        X = features_dict["VLM"][sample_idx]
        Y = features_dict[name][sample_idx]
        acc = compute_1nn_retrieval_accuracy(X, Y)
        pairs.append(f"VLM↔{MODEL_DISPLAY_NAMES[name]}")
        accuracies.append(acc)
        colors.append('#E74C3C')
        print(f"  VLM vs {name}: {acc*100:.2f}%")

    # 再计算 Visual vs Visual
    for i, name_a in enumerate(visual_names):
        for name_b in visual_names[i+1:]:
            X = features_dict[name_a][sample_idx]
            Y = features_dict[name_b][sample_idx]
            acc = compute_1nn_retrieval_accuracy(X, Y)
            pairs.append(f"{MODEL_DISPLAY_NAMES[name_a][:3]}↔{MODEL_DISPLAY_NAMES[name_b][:3]}")
            accuracies.append(acc)
            colors.append('#3498DB')
            print(f"  {name_a} vs {name_b}: {acc*100:.2f}%")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pairs))
    bars = ax.bar(x, [a*100 for a in accuracies], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

    # 先设置 ylim
    max_acc = max(accuracies) * 100
    ax.set_ylim(0, max_acc * 1.12)  # 留 12% 空间给标注

    # 添加数值标签（在条形图顶部内侧）
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max_acc * 0.01,
               f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加随机基线
    random_baseline = 1.0 / n_samples * 100
    ax.axhline(random_baseline, color='gray', linestyle='--', linewidth=2, label=f'Random Baseline ({random_baseline:.2f}%)')

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('1-NN Retrieval Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Model Scene Retrieval Accuracy\n(Can Model A find the same scene in Model B?)',
                fontsize=13, fontweight='bold')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='VLM vs Visual'),
        mpatches.Patch(facecolor='#3498DB', label='Visual vs Visual'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[方案 D2] 完成!")
    print(f"  保存到: {save_path}")


def plot_comprehensive_advanced(
    features_dict: Dict[str, np.ndarray],
    cka_matrix: np.ndarray,
    model_names: List[str],
    save_path: str,
    n_samples: int = 2000,
) -> None:
    """
    综合高级可视化图（4合1论文图）
    """
    print(f"\n[综合高级可视化] 开始生成...")
    start_time = time.time()
    np.random.seed(42)

    fig = plt.figure(figsize=(16, 14))

    # ============ (a) 层次聚类 Dendrogram ============
    ax1 = fig.add_subplot(221)

    distance_matrix = np.sqrt(2 * (1 - cka_matrix))
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method='average')

    display_names = [MODEL_DISPLAY_NAMES[name] for name in model_names]
    dendrogram(Z, labels=display_names, ax=ax1, leaf_font_size=10,
              color_threshold=0, above_threshold_color='#2C3E50')

    ax1.set_ylabel('Distance', fontsize=10)
    ax1.set_title('(a) Hierarchical Clustering (CKA Distance)', fontsize=12, fontweight='bold')

    # ============ (b) CKA 热图 ============
    ax2 = fig.add_subplot(222)

    im = ax2.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            color = 'white' if cka_matrix[i, j] > 0.5 else 'black'
            ax2.text(j, i, f'{cka_matrix[i, j]:.2f}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    ax2.set_xticks(np.arange(len(model_names)))
    ax2.set_yticks(np.arange(len(model_names)))
    ax2.set_xticklabels(display_names, fontsize=9, rotation=30, ha='right')
    ax2.set_yticklabels(display_names, fontsize=9)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title('(b) CKA Similarity Matrix', fontsize=12, fontweight='bold')

    # ============ (c) Matched vs Random 相似度 (VLM vs Visual) ============
    ax3 = fig.add_subplot(223)

    n_total = features_dict[model_names[0]].shape[0]
    n_samples_sim = min(5000, n_total)
    sample_idx = np.random.choice(n_total, n_samples_sim, replace=False)

    # 计算 VLM vs 所有视觉模型
    visual_names = [n for n in model_names if n != "VLM"]

    for name in visual_names:
        X = features_dict["VLM"][sample_idx]
        Y = features_dict[name][sample_idx]
        matched_sim = compute_cosine_similarity(X, Y)
        ax3.hist(matched_sim, bins=40, alpha=0.5, label=f'VLM↔{MODEL_DISPLAY_NAMES[name]}', density=True)

    # Visual vs Visual (选一个代表)
    X = features_dict["ViT-Large"][sample_idx]
    Y = features_dict["EVA-CLIP"][sample_idx]
    matched_sim = compute_cosine_similarity(X, Y)
    ax3.hist(matched_sim, bins=40, alpha=0.5, color='#2C3E50', label='ViT↔EVA (Visual)', density=True)

    ax3.set_xlabel('Cosine Similarity', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('(c) Matched Similarity Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ============ (d) 1-NN 检索准确率 ============
    ax4 = fig.add_subplot(224)

    pairs = []
    accuracies = []
    colors = []

    # VLM vs Visual
    for name in visual_names:
        X = features_dict["VLM"][sample_idx]
        Y = features_dict[name][sample_idx]
        acc = compute_1nn_retrieval_accuracy(X, Y)
        pairs.append(f"VLM↔{name[:3]}")
        accuracies.append(acc)
        colors.append('#E74C3C')

    # Visual vs Visual
    for i, name_a in enumerate(visual_names):
        for name_b in visual_names[i+1:]:
            X = features_dict[name_a][sample_idx]
            Y = features_dict[name_b][sample_idx]
            acc = compute_1nn_retrieval_accuracy(X, Y)
            pairs.append(f"{name_a[:3]}↔{name_b[:3]}")
            accuracies.append(acc)
            colors.append('#3498DB')

    x = np.arange(len(pairs))
    bars = ax4.bar(x, [a*100 for a in accuracies], color=colors, alpha=0.8)

    # 先设置 ylim，再添加标注
    max_acc = max(accuracies) * 100
    ax4.set_ylim(0, max_acc * 1.15)  # 留 15% 空间给标注

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + max_acc * 0.02,
               f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_xticks(x)
    ax4.set_xticklabels(pairs, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Accuracy (%)', fontsize=10)
    ax4.set_title('(d) 1-NN Cross-Model Retrieval', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='VLM vs Visual'),
        mpatches.Patch(facecolor='#3498DB', label='Visual vs Visual'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.suptitle('Multi-Model Representation Space Analysis\n'
                '(VLM shows distinct representation vs Visual Backbone Methods)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[综合高级可视化] 完成! 耗时: {time.time() - start_time:.1f}s")
    print(f"  保存到: {save_path}")


# ============ 方案 B: SAE Shared/Unique 可视化 ============

def load_sae_model(model_path: str, device: torch.device = None):
    """加载训练好的 SAE 模型"""
    if not HAS_TORCH:
        raise ImportError("PyTorch 未安装")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = SharedUniqueSAE(
        dim_x=config['dim_x'],
        dim_y=config['dim_y'],
        shared_dim=config['shared_dim'],
        unique_x_dim=config['unique_x_dim'],
        unique_y_dim=config['unique_y_dim'],
        hidden_dim=config['hidden_dim'],
        use_linear_decoder=config['use_linear_decoder'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.scaler_x = checkpoint['scaler_x']
    model.scaler_y = checkpoint['scaler_y']
    model.eval()

    return model, config


def extract_sae_latents(
    model: 'SharedUniqueSAE',
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """从 SAE 模型提取 shared 和 unique latent codes"""
    if device is None:
        device = next(model.parameters()).device

    # 标准化
    X_scaled = model.scaler_x.transform(X).astype(np.float32)
    Y_scaled = model.scaler_y.transform(Y).astype(np.float32)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        Y_tensor = torch.from_numpy(Y_scaled).to(device)

        shared_x = model.shared_encoder_x(X_tensor).cpu().numpy()
        shared_y = model.shared_encoder_y(Y_tensor).cpu().numpy()
        unique_x = model.unique_encoder_x(X_tensor).cpu().numpy()
        unique_y = model.unique_encoder_y(Y_tensor).cpu().numpy()

    return {
        'shared_x': shared_x,
        'shared_y': shared_y,
        'unique_x': unique_x,
        'unique_y': unique_y,
    }


def plot_sae_shared_unique_visualization(
    features_dict: Dict[str, np.ndarray],
    model_pairs: List[Tuple[str, str]],
    sae_model_dir: str,
    save_path: str,
    n_samples: int = 2000,
    device: torch.device = None,
) -> None:
    """
    方案 B: SAE shared/unique latent space 可视化

    展示：
    - shared latent: 四个模型应高度重叠（公共因子）
    - unique latent: VLM 的 unique 与视觉方法分离
    """
    if not HAS_TORCH:
        print("[方案 B] 跳过: PyTorch 未安装")
        return

    print(f"\n[方案 B] SAE Shared/Unique 可视化")
    start_time = time.time()
    np.random.seed(42)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 采样
    n_total = features_dict[list(features_dict.keys())[0]].shape[0]
    n_samples = min(n_samples, n_total)
    sample_idx = np.random.choice(n_total, n_samples, replace=False)

    # 选择一个代表性的模型对进行可视化（VLM vs ViT-Large）
    # 也可以扩展为多对
    target_pair = ("VLM", "ViT-Large")

    # 构建 SAE 模型路径
    short_names = {"VLM": "VLM", "ViT-Large": "ViT", "ResNet101": "ResNet", "EVA-CLIP": "EVA"}
    model_filename = f"sae_model_{short_names[target_pair[0]]}_{short_names[target_pair[1]]}.pt"
    model_path = os.path.join(sae_model_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"  警告: SAE 模型不存在: {model_path}")
        print(f"  请先运行 analyze_multi_models.py 训练 SAE 模型")
        return

    print(f"  加载 SAE 模型: {model_path}")
    model, config = load_sae_model(model_path, device)

    # 获取特征
    X = features_dict[target_pair[0]][sample_idx]
    Y = features_dict[target_pair[1]][sample_idx]

    # 提取 latent codes
    print(f"  提取 latent codes ({n_samples} 样本)...")
    latents = extract_sae_latents(model, X, Y, device)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ============ (a) Shared Space - PCA ============
    ax = axes[0, 0]

    # 合并 shared
    all_shared = np.vstack([latents['shared_x'], latents['shared_y']])
    pca_shared = PCA(n_components=2)
    shared_coords = pca_shared.fit_transform(all_shared)

    shared_x_coords = shared_coords[:n_samples]
    shared_y_coords = shared_coords[n_samples:]

    ax.scatter(shared_x_coords[:, 0], shared_x_coords[:, 1],
              c=MODEL_COLORS[target_pair[0]], s=8, alpha=0.5, label=f'{target_pair[0]} shared')
    ax.scatter(shared_y_coords[:, 0], shared_y_coords[:, 1],
              c=MODEL_COLORS[target_pair[1]], s=8, alpha=0.5, label=f'{target_pair[1]} shared')

    ax.set_xlabel(f'PC1 ({pca_shared.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca_shared.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.set_title('(a) Shared Latent Space (PCA)\nExpected: High Overlap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ============ (b) Unique Space - PCA ============
    ax = axes[0, 1]

    # unique 维度可能不同，分别做 PCA
    pca_ux = PCA(n_components=2)
    pca_uy = PCA(n_components=2)
    unique_x_coords = pca_ux.fit_transform(latents['unique_x'])
    unique_y_coords = pca_uy.fit_transform(latents['unique_y'])

    ax.scatter(unique_x_coords[:, 0], unique_x_coords[:, 1],
              c=MODEL_COLORS[target_pair[0]], s=8, alpha=0.5, label=f'{target_pair[0]} unique')
    ax.scatter(unique_y_coords[:, 0], unique_y_coords[:, 1],
              c=MODEL_COLORS[target_pair[1]], s=8, alpha=0.5, label=f'{target_pair[1]} unique')

    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_title('(b) Unique Latent Space (PCA)\nExpected: Separation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ============ (c) Shared 对齐质量 ============
    ax = axes[1, 0]

    # 计算 shared 的逐样本余弦相似度
    shared_x_norm = latents['shared_x'] / (np.linalg.norm(latents['shared_x'], axis=1, keepdims=True) + 1e-8)
    shared_y_norm = latents['shared_y'] / (np.linalg.norm(latents['shared_y'], axis=1, keepdims=True) + 1e-8)
    shared_cosine = np.sum(shared_x_norm * shared_y_norm, axis=1)

    ax.hist(shared_cosine, bins=50, color=MODEL_COLORS[target_pair[0]], alpha=0.7, edgecolor='white')
    ax.axvline(shared_cosine.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {shared_cosine.mean():.3f}')

    ax.set_xlabel('Cosine Similarity (Shared)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(c) Shared Code Alignment Quality\n(Same Scene, Cross-Model)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ============ (d) Unique 能量分布 ============
    ax = axes[1, 1]

    # 计算 unique 的 L2 范数（能量）
    unique_x_energy = np.linalg.norm(latents['unique_x'], axis=1)
    unique_y_energy = np.linalg.norm(latents['unique_y'], axis=1)

    ax.hist(unique_x_energy, bins=50, alpha=0.6, color=MODEL_COLORS[target_pair[0]],
           label=f'{target_pair[0]} unique (mean={unique_x_energy.mean():.2f})')
    ax.hist(unique_y_energy, bins=50, alpha=0.6, color=MODEL_COLORS[target_pair[1]],
           label=f'{target_pair[1]} unique (mean={unique_y_energy.mean():.2f})')

    ax.set_xlabel('L2 Norm (Unique Code)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(d) Unique Code Energy Distribution\n(Higher = More Model-Specific Info)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'SAE Shared/Unique Decomposition: {target_pair[0]} vs {target_pair[1]}\n'
                f'(Shared dim={config["shared_dim"]}, Unique dim={config["unique_x_dim"]}/{config["unique_y_dim"]})',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 打印统计信息
    print(f"\n  [统计信息]")
    print(f"    Shared 余弦相似度: mean={shared_cosine.mean():.4f}, std={shared_cosine.std():.4f}")
    print(f"    Unique X 能量: mean={unique_x_energy.mean():.4f}, std={unique_x_energy.std():.4f}")
    print(f"    Unique Y 能量: mean={unique_y_energy.mean():.4f}, std={unique_y_energy.std():.4f}")

    print(f"\n[方案 B] 完成! 耗时: {time.time() - start_time:.1f}s")
    print(f"  保存到: {save_path}")


def plot_backbone_vs_dit_comparison(
    features_backbone: Dict[str, np.ndarray],
    features_dit: Dict[str, np.ndarray],
    model_names: List[str],
    save_path: str,
    reference_model: str = "ResNet101",
    n_samples: int = 1500,
) -> None:
    """
    并排对比 Backbone 和 DiT 特征的 UMAP 可视化
    """
    print(f"\n[Backbone vs DiT 对比] 开始生成...")
    start_time = time.time()
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    feature_types = [
        ('Backbone', features_backbone),
        ('DiT', features_dit),
    ]

    for ax_idx, (feat_name, features_dict) in enumerate(feature_types):
        ax = axes[ax_idx]

        # Procrustes 对齐
        aligned = align_to_reference(features_dict, model_names, reference_model)

        # 采样
        n_total = features_dict[model_names[0]].shape[0]
        n_samples_actual = min(n_samples, n_total)
        sample_idx = np.random.choice(n_total, n_samples_actual, replace=False)

        # 合并对齐后的特征
        all_aligned = np.vstack([aligned[name][sample_idx] for name in model_names])

        # UMAP 或 PCA
        print(f"  [{feat_name}] 运行降维...")
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42, verbose=False)
            coords = reducer.fit_transform(all_aligned)
        else:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(all_aligned)

        # 分离每个模型的坐标
        model_coords = {}
        start_idx = 0
        for name in model_names:
            end_idx = start_idx + n_samples_actual
            model_coords[name] = coords[start_idx:end_idx]
            start_idx = end_idx

        # 绘制点云
        for name in model_names:
            ax.scatter(model_coords[name][:, 0], model_coords[name][:, 1],
                      c=MODEL_COLORS[name], s=10, alpha=0.6,
                      label=MODEL_DISPLAY_NAMES[name])

        # 绘制每个模型的中心点
        for name in model_names:
            center = model_coords[name].mean(axis=0)
            ax.scatter(center[0], center[1], c=MODEL_COLORS[name],
                      s=300, marker='X', edgecolors='white', linewidths=2, zorder=10)

        method = "UMAP" if HAS_UMAP else "PCA"
        ax.set_xlabel(f'{method} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=11)
        ax.set_title(f'{feat_name} Features\n(Procrustes-aligned to {reference_model})',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, markerscale=1.5)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Multi-Model Feature Space: Backbone vs DiT Layer Comparison',
                fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[Backbone vs DiT 对比] 完成! 耗时: {time.time() - start_time:.1f}s")
    print(f"  保存到: {save_path}")


def run_analysis_for_feature_type(
    features_dict: Dict[str, np.ndarray],
    model_names: List[str],
    output_dir: str,
    feature_type: str,  # "backbone" or "dit"
    reference_model: str = "ResNet101",
) -> np.ndarray:
    """
    对单一特征类型运行完整分析流程

    返回: CKA 矩阵
    """
    print(f"\n{'='*80}")
    print(f"分析 {feature_type.upper()} 特征")
    print(f"{'='*80}")

    suffix = f"_{feature_type}"

    # 计算 CKA
    print(f"\n计算 {feature_type} CKA 矩阵...")
    cka_matrix = np.eye(len(model_names))
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            cka_val = linear_cka(features_dict[model_names[i]], features_dict[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val

    print(f"\n{feature_type} CKA 矩阵:")
    for i, name in enumerate(model_names):
        print(f"  {name:<12}" + "".join([f"{cka_matrix[i,j]:>8.4f}" for j in range(len(model_names))]))

    # 方案 A: Procrustes + UMAP
    plot_procrustes_aligned_umap(
        features_dict, model_names,
        os.path.join(output_dir, f"procrustes_aligned_umap{suffix}.png"),
        reference_model=reference_model,
        n_samples=2000,
        show_arrows=True,
        arrow_sample_ratio=0.03,
    )

    # 方案 C: 层次聚类
    plot_hierarchical_clustering(
        cka_matrix, model_names,
        os.path.join(output_dir, f"hierarchical_clustering{suffix}.png"),
    )

    # 方案 D1: Matched vs Random 分布
    plot_similarity_distribution(
        features_dict, model_names,
        os.path.join(output_dir, f"similarity_distribution{suffix}.png"),
        n_samples=5000,
    )

    # 方案 D2: 1-NN 检索准确率
    plot_retrieval_accuracy(
        features_dict, model_names,
        os.path.join(output_dir, f"retrieval_accuracy{suffix}.png"),
        n_samples=5000,
    )

    # 综合图
    plot_comprehensive_advanced(
        features_dict, cka_matrix, model_names,
        os.path.join(output_dir, f"comprehensive_advanced{suffix}.png"),
        n_samples=2000,
    )

    return cka_matrix


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize archived multi-model representations.")
    parser.add_argument("--cache-dir", required=True, help="Directory of aligned multi-model NPZs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", help="Trusted SAE checkpoint directory for latent plots.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    output_dir = args.output
    cache_dir = args.cache_dir
    model_names = ["VLM", "ViT-Large", "ResNet101", "EVA-CLIP"]

    print("=" * 80)
    print("多模型表征空间高级可视化")
    print("=" * 80)

    # ============ 加载 Backbone 特征 ============
    cache_path_backbone = os.path.join(cache_dir, "all_models_aligned_cache.npz")
    features_backbone = None

    if os.path.exists(cache_path_backbone):
        print(f"\n[Backbone] 从缓存加载: {cache_path_backbone}")
        cache_data = np.load(cache_path_backbone, allow_pickle=True)
        features_backbone = {name: cache_data[name] for name in model_names}
        print(f"  已加载 {features_backbone[model_names[0]].shape[0]} 个场景")
    else:
        print(f"\n[Backbone] 警告: 缓存不存在 {cache_path_backbone}")

    # ============ 加载 DiT 特征 ============
    cache_path_dit = os.path.join(cache_dir, "all_models_aligned_cache_dit.npz")
    features_dit = None

    if os.path.exists(cache_path_dit):
        print(f"\n[DiT] 从缓存加载: {cache_path_dit}")
        cache_data = np.load(cache_path_dit, allow_pickle=True)
        features_dit = {name: cache_data[name] for name in model_names}
        print(f"  已加载 {features_dit[model_names[0]].shape[0]} 个场景")
    else:
        print(f"\n[DiT] 警告: 缓存不存在 {cache_path_dit}")
        print(f"  如需生成 DiT 缓存，请先运行 visualize_multi_model_space.py 生成")

    if features_backbone is None and features_dit is None:
        print("\n错误: 没有找到任何特征缓存文件，请先运行 visualize_multi_model_space.py")
        return

    # ============ 运行分析 ============
    cka_backbone = None
    cka_dit = None

    # Backbone 特征分析
    if features_backbone is not None:
        cka_backbone = run_analysis_for_feature_type(
            features_backbone, model_names, output_dir,
            feature_type="backbone",
            reference_model="ResNet101",
        )

        # SAE 可视化 (仅 Backbone)
        if HAS_TORCH and args.model_dir:
            model_pairs = [("VLM", "ViT-Large")]
            plot_sae_shared_unique_visualization(
                features_backbone, model_pairs, args.model_dir,
                os.path.join(output_dir, "sae_shared_unique_backbone.png"),
                n_samples=2000,
            )

    # DiT 特征分析
    if features_dit is not None:
        cka_dit = run_analysis_for_feature_type(
            features_dit, model_names, output_dir,
            feature_type="dit",
            reference_model="ResNet101",
        )

    # ============ Backbone vs DiT 对比图 ============
    if features_backbone is not None and features_dit is not None:
        plot_backbone_vs_dit_comparison(
            features_backbone, features_dit, model_names,
            os.path.join(output_dir, "backbone_vs_dit_comparison.png"),
            reference_model="ResNet101",
            n_samples=1500,
        )

        # CKA 差异分析
        print(f"\n{'='*80}")
        print("Backbone vs DiT CKA 差异分析")
        print(f"{'='*80}")
        print("\nCKA 变化 (DiT - Backbone):")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                diff = cka_dit[i, j] - cka_backbone[i, j]
                arrow = "↑" if diff > 0 else "↓"
                print(f"  {model_names[i]:<12} - {model_names[j]:<12}: "
                      f"{cka_backbone[i,j]:.4f} -> {cka_dit[i,j]:.4f} ({arrow}{abs(diff):.4f})")

    # ============ 完成 ============
    print("\n" + "=" * 80)
    print("完成！输出文件:")
    print("=" * 80)

    if features_backbone is not None:
        print("\n[Backbone 特征]:")
        print(f"  - procrustes_aligned_umap_backbone.png")
        print(f"  - hierarchical_clustering_backbone.png")
        print(f"  - similarity_distribution_backbone.png")
        print(f"  - retrieval_accuracy_backbone.png")
        print(f"  - comprehensive_advanced_backbone.png")
        if HAS_TORCH:
            print(f"  - sae_shared_unique_backbone.png")

    if features_dit is not None:
        print("\n[DiT 特征]:")
        print(f"  - procrustes_aligned_umap_dit.png")
        print(f"  - hierarchical_clustering_dit.png")
        print(f"  - similarity_distribution_dit.png")
        print(f"  - retrieval_accuracy_dit.png")
        print(f"  - comprehensive_advanced_dit.png")

    if features_backbone is not None and features_dit is not None:
        print("\n[对比图]:")
        print(f"  - backbone_vs_dit_comparison.png")

    print(f"\n所有文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
