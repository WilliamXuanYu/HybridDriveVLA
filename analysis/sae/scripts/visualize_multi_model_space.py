# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""
多模型表征空间统一可视化脚本

将四个模型的表征空间放在同一个图中进行可视化，
直观展示 VLM 与视觉端到端方法（ViT、ResNet、EVA-CLIP）之间的差异。
"""

import os
import numpy as np

# 在导入 pyplot 之前设置非交互式后端（解决服务器无显示器问题）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from analyze_pca_whitening_cca import (
    load_features_from_dir,
    linear_cka,
)

# ============ 配色方案 ============
MODEL_COLORS = {
    "VLM": "#E74C3C",       # 红色 - VLM 突出显示
    "ViT-Large": "#3498DB", # 蓝色
    "ResNet101": "#2ECC71", # 绿色
    "EVA-CLIP": "#9B59B6",  # 紫色
}

MODEL_MARKERS = {
    "VLM": "o",
    "ViT-Large": "s",
    "ResNet101": "^",
    "EVA-CLIP": "D",
}

MODEL_DISPLAY_NAMES = {
    "VLM": "VLM (RecogDrive)",
    "ViT-Large": "ViT-Large",
    "ResNet101": "ResNet-101",
    "EVA-CLIP": "EVA-CLIP",
}

# Backbone 特征配置
MODEL_CONFIGS_BACKBONE = {
    "VLM": {
        "feature_key": None,
    },
    "ViT-Large": {
        "feature_key": None,
    },
    "ResNet101": {
        "feature_key": None,
    },
    "EVA-CLIP": {
        "feature_key": None,
    },
}

# DiT 特征配置
MODEL_CONFIGS_DIT = {
    "VLM": {
        "feature_key": None,
    },
    "ViT-Large": {
        "feature_key": None,
    },
    "ResNet101": {
        "feature_key": None,
    },
    "EVA-CLIP": {
        "feature_key": None,
    },
}

# 默认使用 Backbone 配置（保持向后兼容）
MODEL_CONFIGS = MODEL_CONFIGS_BACKBONE


def load_all_features(model_names, max_scenes=None, cache_dir=None, feature_type="backbone"):
    """
    加载所有模型的特征，并找到共同的场景ID

    Args:
        model_names: 模型名称列表
        max_scenes: 最大场景数限制
        cache_dir: 缓存目录
        feature_type: 特征类型，"backbone" 或 "dit"

    Returns:
        features_dict: 特征字典
        common_ids: 共同场景ID列表
    """
    # 选择配置
    if feature_type == "dit":
        model_configs = MODEL_CONFIGS_DIT
        cache_suffix = "_dit"
    else:
        model_configs = MODEL_CONFIGS_BACKBONE
        cache_suffix = ""

    # 尝试从缓存加载
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f"all_models_aligned_cache{cache_suffix}.npz")
        if os.path.exists(cache_path):
            print(f"从缓存加载 ({feature_type}): {cache_path}")
            with np.load(cache_path, allow_pickle=True) as cache_data:
                features_dict = {name: cache_data[name] for name in model_names}
                common_ids = cache_data['common_ids'].tolist()
            print(f"已加载 {len(common_ids)} 个共同场景")
            return features_dict, common_ids

    # 加载原始特征
    all_feats = {}
    for name in model_names:
        print(f"加载 {name} ({feature_type}) 特征...")
        config = model_configs[name]

        # 检查路径是否存在
        if not os.path.exists(config["path"]):
            print(f"  警告: 路径不存在 {config['path']}")
            return None, None

        feats = load_features_from_dir(config["path"], feature_key=config["feature_key"])
        all_feats[name] = feats
        print(f"  {name}: {len(feats)} 个场景")

    # 找共同场景
    common_ids = set(all_feats[model_names[0]].keys())
    for name in model_names[1:]:
        common_ids &= set(all_feats[name].keys())
    common_ids = sorted(list(common_ids))

    if max_scenes is not None:
        common_ids = common_ids[:max_scenes]

    if len(common_ids) < 2:
        raise ValueError("At least two scenes must exist in every model export.")
    print(f"\n共同场景数: {len(common_ids)}")

    # 构建特征矩阵
    features_dict = {}
    for name in model_names:
        sample = all_feats[name][common_ids[0]]
        feat_dim = sample.shape[0]
        features = np.zeros((len(common_ids), feat_dim), dtype=np.float32)
        for i, sid in enumerate(tqdm(common_ids, desc=f"构建 {name} 矩阵")):
            features[i] = all_feats[name][sid]
        features_dict[name] = features
        print(f"  {name}: shape = {features.shape}")

    # 保存缓存
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"all_models_aligned_cache{cache_suffix}.npz")
        np.savez_compressed(cache_path, **features_dict, common_ids=np.array(common_ids, dtype=object))
        print(f"缓存已保存: {cache_path}")

    return features_dict, common_ids


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


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """绘制置信椭圆"""
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_comprehensive_figure(cka_matrix, features_dict, model_names, save_path, n_samples=2000):
    """综合可视化图：CKA热图、MDS模型关系图、t-SNE分布图、PCA分布图"""
    import time
    np.random.seed(42)

    print(f"\n[综合可视化] 开始生成 (每个模型采样 {n_samples} 个样本, 总计 {n_samples * len(model_names)} 个)")
    total_start = time.time()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    display_names = [MODEL_DISPLAY_NAMES[name] for name in model_names]

    # (a) CKA 热图
    print("  [1/4] 绘制 CKA 热图...")
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            color = 'white' if cka_matrix[i, j] > 0.5 else 'black'
            ax1.text(j, i, f'{cka_matrix[i, j]:.2f}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    ax1.set_xticks(np.arange(len(model_names)))
    ax1.set_yticks(np.arange(len(model_names)))
    ax1.set_xticklabels(display_names, fontsize=10, rotation=30, ha='right')
    ax1.set_yticklabels(display_names, fontsize=10)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('CKA Similarity', fontsize=10)
    ax1.set_title('(a) CKA Similarity Matrix', fontsize=13, fontweight='bold', pad=10)

    # (b) MDS 模型关系图
    print("  [2/4] 计算 MDS 并绘制模型关系图...")
    ax2 = fig.add_subplot(gs[0, 1])
    distance_matrix = np.sqrt(2 * np.clip(1 - cka_matrix, 0, None))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(distance_matrix)

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            x = [mds_coords[i, 0], mds_coords[j, 0]]
            y = [mds_coords[i, 1], mds_coords[j, 1]]
            cka_val = cka_matrix[i, j]
            ax2.plot(x, y, 'gray', linewidth=1 + cka_val * 3, alpha=0.3 + cka_val * 0.5, zorder=1)
            mid_x, mid_y = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            ax2.annotate(f'{cka_val:.2f}', (mid_x, mid_y), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.7))

    for i, name in enumerate(model_names):
        ax2.scatter(mds_coords[i, 0], mds_coords[i, 1], c=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                   s=350, edgecolors='white', linewidths=2, zorder=2, label=MODEL_DISPLAY_NAMES[name])
        ax2.annotate(MODEL_DISPLAY_NAMES[name], (mds_coords[i, 0], mds_coords[i, 1] + 0.03),
                    fontsize=9, fontweight='bold', ha='center', color=MODEL_COLORS[name])

    ax2.set_xlabel('MDS Dimension 1', fontsize=10)
    ax2.set_ylabel('MDS Dimension 2', fontsize=10)
    ax2.set_title('(b) Model Relationship (MDS)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    # 准备数据
    n_total = features_dict[model_names[0]].shape[0]
    n_samples = min(n_samples, n_total)
    sample_idx = np.random.choice(n_total, n_samples, replace=False)

    all_features = np.vstack([features_dict[name][sample_idx] for name in model_names])
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # (c) t-SNE
    ax3 = fig.add_subplot(gs[1, 0])
    print(f"  [3/4] 运行 t-SNE ({all_features_scaled.shape[0]} 样本)... 这可能需要几分钟，请耐心等待")
    tsne_start = time.time()
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_features_scaled) - 1), random_state=42, max_iter=1000, verbose=1)
    tsne_coords = tsne.fit_transform(all_features_scaled)
    print(f"        t-SNE 完成! 耗时: {time.time() - tsne_start:.1f} 秒")

    start_idx = 0
    for name in model_names:
        end_idx = start_idx + n_samples
        ax3.scatter(tsne_coords[start_idx:end_idx, 0], tsne_coords[start_idx:end_idx, 1],
                   c=MODEL_COLORS[name], marker=MODEL_MARKERS[name], s=10, alpha=0.5, label=MODEL_DISPLAY_NAMES[name])
        start_idx = end_idx

    ax3.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax3.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax3.set_title('(c) Feature Distribution (t-SNE)', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9, markerscale=2)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # (d) PCA
    ax4 = fig.add_subplot(gs[1, 1])
    print("  [4/4] 运行 PCA 并绘制分布图...")
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_features_scaled)

    start_idx = 0
    for name in model_names:
        end_idx = start_idx + n_samples
        model_coords = pca_coords[start_idx:end_idx]
        ax4.scatter(model_coords[:, 0], model_coords[:, 1], c=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                   s=10, alpha=0.5, label=MODEL_DISPLAY_NAMES[name])
        try:
            confidence_ellipse(model_coords[:, 0], model_coords[:, 1], ax4, n_std=2.0,
                              edgecolor=MODEL_COLORS[name], linewidth=2, linestyle='-', alpha=0.8)
        except:
            pass
        start_idx = end_idx

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax4.set_title('(d) Feature Distribution (PCA)', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9, markerscale=2)
    ax4.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Multi-Model Representation Space Comparison', fontsize=16, fontweight='bold', y=0.98)
    print("  保存图片...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[综合可视化] 完成! 总耗时: {time.time() - total_start:.1f} 秒")
    print(f"  保存到: {save_path}")


def plot_vlm_vs_visual_comparison(cka_matrix, features_dict, model_names, save_path, n_samples=3000):
    """专门对比 VLM 与视觉端到端方法的差异 - 适合作为论文核心图"""
    import time
    print(f"\n[VLM vs Visual 对比图] 开始生成...")
    start_time = time.time()
    np.random.seed(42)
    fig = plt.figure(figsize=(14, 6))

    # 左图: MDS 模型关系图
    ax1 = fig.add_subplot(121)
    distance_matrix = np.sqrt(2 * np.clip(1 - cka_matrix, 0, None))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(distance_matrix)

    vlm_idx = model_names.index("VLM")
    visual_idx = [i for i in range(len(model_names)) if model_names[i] != "VLM"]

    # 视觉方法之间的强连接
    for i in visual_idx:
        for j in visual_idx:
            if i < j:
                x = [mds_coords[i, 0], mds_coords[j, 0]]
                y = [mds_coords[i, 1], mds_coords[j, 1]]
                cka_val = cka_matrix[i, j]
                ax1.plot(x, y, color='#2C3E50', linewidth=2 + cka_val * 3, alpha=0.6, zorder=1)

    # VLM 到视觉方法的弱连接
    for j in visual_idx:
        x = [mds_coords[vlm_idx, 0], mds_coords[j, 0]]
        y = [mds_coords[vlm_idx, 1], mds_coords[j, 1]]
        ax1.plot(x, y, color='#E74C3C', linewidth=1.5, alpha=0.5, linestyle='--', zorder=1)

    # 视觉方法区域
    visual_coords = mds_coords[visual_idx]
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(visual_coords)
        hull_points = np.vstack([visual_coords[hull.vertices], visual_coords[hull.vertices[0]]])
        ax1.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.15, color='#3498DB', label='Visual Backbone Region')
    except:
        pass

    # 绘制模型点
    for i, name in enumerate(model_names):
        size = 500 if name == "VLM" else 400
        ax1.scatter(mds_coords[i, 0], mds_coords[i, 1], c=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                   s=size, edgecolors='white', linewidths=2.5, zorder=3, label=MODEL_DISPLAY_NAMES[name])

    # CKA 标注
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            cka_val = cka_matrix[i, j]
            mid_x = (mds_coords[i, 0] + mds_coords[j, 0]) / 2
            mid_y = (mds_coords[i, 1] + mds_coords[j, 1]) / 2
            ax1.annotate(f'{cka_val:.2f}', (mid_x + 0.01, mid_y + 0.01), fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.85))

    ax1.set_xlabel('Representation Distance', fontsize=11)
    ax1.set_ylabel('Representation Distance', fontsize=11)
    ax1.set_title('(a) Model Representation Similarity\n(MDS Embedding based on CKA)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')

    # 右图: CKA 条形图
    ax2 = fig.add_subplot(122)

    vlm_cka_values = [(model_names[j], cka_matrix[vlm_idx, j]) for j in visual_idx]
    vlm_cka_values.sort(key=lambda x: x[1], reverse=True)

    visual_cka_values = []
    for i in visual_idx:
        for j in visual_idx:
            if i < j:
                visual_cka_values.append((f"{model_names[i][:3]}-{model_names[j][:3]}", cka_matrix[i, j]))
    visual_cka_values.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(vlm_cka_values) + len(visual_cka_values) + 1)

    ax2.barh(y_pos[:len(vlm_cka_values)], [v[1] for v in vlm_cka_values], color='#E74C3C', alpha=0.8, label='VLM vs Visual')
    ax2.barh(y_pos[len(vlm_cka_values)+1:], [v[1] for v in visual_cka_values], color='#3498DB', alpha=0.8, label='Visual vs Visual')

    labels = [f"VLM vs {MODEL_DISPLAY_NAMES[v[0]]}" for v in vlm_cka_values]
    labels.append("")
    labels.extend([f"{v[0]}" for v in visual_cka_values])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)

    for i, (name, val) in enumerate(vlm_cka_values):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    for i, (name, val) in enumerate(visual_cka_values):
        ax2.text(val + 0.02, len(vlm_cka_values) + 1 + i, f'{val:.3f}', va='center', fontsize=10)

    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('CKA Similarity', fontsize=11)
    ax2.set_title('(b) CKA Similarity Comparison\nVLM vs Visual Backbone Methods', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=len(vlm_cka_values), color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VLM vs Visual 对比图] 完成! 耗时: {time.time() - start_time:.1f} 秒")
    print(f"  保存到: {save_path}")


def run_analysis_for_feature_type(
    model_names, cache_dir, output_dir, feature_type, n_samples_viz=4000
):
    """对单一特征类型运行分析"""
    suffix = f"_{feature_type}" if feature_type != "backbone" else ""

    print(f"\n{'='*80}")
    print(f"处理 {feature_type.upper()} 特征")
    print(f"{'='*80}")

    # 加载特征
    features_dict, common_ids = load_all_features(
        model_names, cache_dir=cache_dir, feature_type=feature_type
    )

    if features_dict is None:
        print(f"跳过 {feature_type}: 特征加载失败")
        return None, None

    # 计算 CKA
    print(f"\n计算 {feature_type} CKA 矩阵...")
    cka_matrix = compute_cka_matrix(features_dict, model_names)
    print(f"\n{feature_type} CKA 矩阵:")
    for i, name in enumerate(model_names):
        row = f"  {name:<12}"
        for j in range(len(model_names)):
            row += f"{cka_matrix[i, j]:>8.4f}"
        print(row)

    # 生成可视化
    print(f"\n生成 {feature_type} 可视化...")

    plot_comprehensive_figure(
        cka_matrix, features_dict, model_names,
        os.path.join(output_dir, f"comprehensive_comparison{suffix}.png"),
        n_samples=n_samples_viz
    )

    plot_vlm_vs_visual_comparison(
        cka_matrix, features_dict, model_names,
        os.path.join(output_dir, f"vlm_vs_visual_comparison{suffix}.png"),
        n_samples=n_samples_viz
    )

    return features_dict, cka_matrix


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='多模型表征空间统一可视化')
    parser.add_argument('--feature-type', type=str, default='both',
                        choices=['backbone', 'dit', 'both'],
                        help='特征类型: backbone, dit, 或 both')
    parser.add_argument('--n-samples', type=int, default=4000,
                        help='可视化采样数量')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-dir', help='Defaults to OUTPUT/aligned_cache.')
    parser.add_argument('--model', action='append', nargs=2, metavar=('NAME', 'EXPORT_ROOT'),
                        help='Supply the feature export root for each of the four models when building caches.')
    args = parser.parse_args()
    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2.")

    output_dir = args.output
    cache_dir = args.cache_dir or os.path.join(output_dir, "aligned_cache")
    model_names = ["VLM", "ViT-Large", "ResNet101", "EVA-CLIP"]
    directories = dict(args.model or [])
    if directories and (set(directories) != set(model_names) or len(args.model) != len(model_names)):
        parser.error(f"Supply one --model per name: {', '.join(model_names)}.")
    for name, directory in directories.items():
        MODEL_CONFIGS_BACKBONE[name]["path"] = os.path.join(directory, "recogdrive_mapped_features")
        MODEL_CONFIGS_DIT[name]["path"] = os.path.join(directory, "recogdrive_dit_output")
    suffixes = {"backbone": "", "dit": "_dit"}
    types = list(suffixes) if args.feature_type == "both" else [args.feature_type]
    if not directories:
        for feature_type in types:
            cache = os.path.join(cache_dir, f"all_models_aligned_cache{suffixes[feature_type]}.npz")
            if not os.path.isfile(cache):
                parser.error(f"{cache} is missing. Supply --model export roots to build it.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 80)
    print("多模型表征空间统一可视化")
    print(f"特征类型: {args.feature_type}")
    print("=" * 80)

    results = {}

    # 处理 Backbone 特征
    if args.feature_type in ['backbone', 'both']:
        features_backbone, cka_backbone = run_analysis_for_feature_type(
            model_names, cache_dir, output_dir, 'backbone', args.n_samples
        )
        if features_backbone is not None:
            results['backbone'] = {'features': features_backbone, 'cka': cka_backbone}

    # 处理 DiT 特征
    if args.feature_type in ['dit', 'both']:
        features_dit, cka_dit = run_analysis_for_feature_type(
            model_names, cache_dir, output_dir, 'dit', args.n_samples
        )
        if features_dit is not None:
            results['dit'] = {'features': features_dit, 'cka': cka_dit}

    # 如果两种特征都有，打印 CKA 差异
    if 'backbone' in results and 'dit' in results:
        print(f"\n{'='*80}")
        print("Backbone vs DiT CKA 差异分析")
        print(f"{'='*80}")
        print("\nCKA 变化 (DiT - Backbone):")
        cka_backbone = results['backbone']['cka']
        cka_dit = results['dit']['cka']
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                diff = cka_dit[i, j] - cka_backbone[i, j]
                arrow = "↑" if diff > 0 else "↓"
                print(f"  {model_names[i]:<12} - {model_names[j]:<12}: "
                      f"{cka_backbone[i,j]:.4f} -> {cka_dit[i,j]:.4f} ({arrow}{abs(diff):.4f})")

    # 输出文件列表
    print("\n" + "=" * 80)
    print("完成！输出文件:")
    print("=" * 80)

    if 'backbone' in results:
        print("\n[Backbone 特征]:")
        print(f"  - comprehensive_comparison.png")
        print(f"  - vlm_vs_visual_comparison.png")

    if 'dit' in results:
        print("\n[DiT 特征]:")
        print(f"  - comprehensive_comparison_dit.png")
        print(f"  - vlm_vs_visual_comparison_dit.png")

    print(f"\n缓存文件保存在: {cache_dir}")
    print(f"可视化文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
