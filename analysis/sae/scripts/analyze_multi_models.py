# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""
多模型表征空间分析脚本

分析四个模型的两两表征空间相似性，包括：
1. CKA 相似度矩阵
2. PCA + CCA 分析
3. SAE 共享/特有子空间分析

模型列表：
- VLM (RecogDrive)
- ViT-Large
- ResNet101
- EVA-CLIP
"""

import os
import pickle
import itertools
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None
    TensorDataset = None

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 复用原有代码中的函数
from analyze_pca_whitening_cca import (
    load_single_feature_from_obj,
    load_features_from_dir,
    align_features,
    center_features,
    linear_cka,
    pca_whitening_cca_analysis,
)

from analyze_sae_shared_unique import (
    SharedUniqueSAE,
    train_sae_model,
    sae_analysis,
)


# ============ 模型配置 ============
MODEL_CONFIGS = {
    "VLM": {
        "feature_key": None,
        "display_name": "VLM (RecogDrive)",
        "short_name": "VLM",
    },
    "ViT-Large": {
        "feature_key": None,
        "display_name": "ViT-Large",
        "short_name": "ViT",
    },
    "ResNet101": {
        "feature_key": None,
        "display_name": "ResNet-101",
        "short_name": "ResNet",
    },
    "EVA-CLIP": {
        "feature_key": None,
        "display_name": "EVA-CLIP",
        "short_name": "EVA",
    },
}


def get_cache_dir(output_base_dir: str) -> str:
    """获取缓存目录"""
    cache_dir = os.path.join(output_base_dir, "aligned_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_path(cache_dir: str, model_a: str, model_b: str) -> str:
    """获取两个模型对齐后的缓存文件路径"""
    # 保证顺序一致
    if model_a > model_b:
        model_a, model_b = model_b, model_a
    return os.path.join(cache_dir, f"aligned_{model_a}_{model_b}.npz")


def load_or_create_aligned_cache(
    model_a: str,
    model_b: str,
    cache_dir: str,
    max_scenes: Optional[int] = None,
    force_reload: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, str]:
    """
    加载或创建两个模型的对齐缓存

    返回: (X, Y, common_ids, actual_model_a, actual_model_b)
    其中 actual_model_a, actual_model_b 是排序后的模型名
    """
    # 保证顺序一致（字典序）
    if model_a > model_b:
        model_a, model_b = model_b, model_a
        swapped = True
    else:
        swapped = False

    cache_path = get_cache_path(cache_dir, model_a, model_b)

    if not force_reload and os.path.exists(cache_path):
        print(f"  从缓存加载 [{model_a}] vs [{model_b}]: {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True)
        X = cache_data['X']
        Y = cache_data['Y']
        common_ids = cache_data['common_ids'].tolist()
        print(f"    样本数: {X.shape[0]}, X 维度: {X.shape[1]}, Y 维度: {Y.shape[1]}")
    else:
        print(f"  创建缓存 [{model_a}] vs [{model_b}]...")

        # 加载特征
        config_a = MODEL_CONFIGS[model_a]
        config_b = MODEL_CONFIGS[model_b]

        print(f"    加载 {model_a} 特征...")
        feats_a = load_features_from_dir(config_a["path"], feature_key=config_a["feature_key"])

        print(f"    加载 {model_b} 特征...")
        feats_b = load_features_from_dir(config_b["path"], feature_key=config_b["feature_key"])

        # 对齐
        print(f"    对齐特征...")
        X, Y, common_ids = align_features(feats_a, feats_b, max_scenes=max_scenes)

        # 保存缓存
        print(f"    保存缓存到: {cache_path}")
        np.savez_compressed(
            cache_path,
            X=X,
            Y=Y,
            common_ids=np.array(common_ids, dtype=object),
            model_a=model_a,
            model_b=model_b,
        )
        print(f"    缓存大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")

    return X, Y, common_ids, model_a, model_b


def compute_cka_matrix(
    model_names: List[str],
    cache_dir: str,
    features_cache: Dict[str, np.ndarray],
    max_scenes: Optional[int] = None,
) -> np.ndarray:
    """
    计算所有模型对的 CKA 矩阵

    参数:
    - model_names: 模型名列表
    - cache_dir: 缓存目录
    - features_cache: 已加载的特征缓存 {model_name: features}
    - max_scenes: 最大场景数

    返回: CKA 矩阵 (n_models, n_models)
    """
    n_models = len(model_names)
    cka_matrix = np.eye(n_models)  # 对角线为 1

    print("\n计算 CKA 矩阵...")

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a, model_b = model_names[i], model_names[j]

            # 加载对齐的特征
            X, Y, _, actual_a, actual_b = load_or_create_aligned_cache(
                model_a, model_b, cache_dir, max_scenes=max_scenes
            )

            # 如果顺序被交换了，需要调整
            if actual_a != model_a:
                X, Y = Y, X

            # 计算 CKA
            cka_val = linear_cka(X, Y)
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            print(f"  CKA({model_a}, {model_b}) = {cka_val:.6f}")

    return cka_matrix


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    model_names: List[str],
    save_path: str,
    title: str = "CKA Similarity Matrix",
) -> None:
    """绘制 CKA 热图"""
    n_models = len(model_names)

    # 获取显示名称
    display_names = [MODEL_CONFIGS[name]["display_name"] for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热图
    im = ax.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)

    # 添加数值标注
    for i in range(n_models):
        for j in range(n_models):
            color = 'white' if cka_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cka_matrix[i, j]:.3f}', ha='center', va='center',
                   color=color, fontsize=12, fontweight='bold')

    # 设置刻度
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(display_names, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(display_names, fontsize=11)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('CKA Similarity', fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  CKA 热图已保存到: {save_path}")


def run_pairwise_cca_analysis(
    model_pairs: List[Tuple[str, str]],
    cache_dir: str,
    output_dir: str,
    pca_variance: float = 0.99,
    cca_threshold: float = 0.8,
    max_scenes: Optional[int] = None,
) -> Dict[Tuple[str, str], Dict]:
    """
    对所有模型对运行 PCA + CCA 分析

    返回: {(model_a, model_b): analysis_results}
    """
    results = {}

    print("\n" + "=" * 80)
    print("PCA + CCA 分析")
    print("=" * 80)

    for model_a, model_b in model_pairs:
        print(f"\n{'-' * 60}")
        print(f"分析: {model_a} vs {model_b}")
        print(f"{'-' * 60}")

        # 加载对齐的特征
        X, Y, common_ids, actual_a, actual_b = load_or_create_aligned_cache(
            model_a, model_b, cache_dir, max_scenes=max_scenes
        )

        # 如果顺序被交换了，需要调整
        if actual_a != model_a:
            X, Y = Y, X
            actual_a, actual_b = model_a, model_b

        # 生成谱图路径
        spectrum_path = os.path.join(
            output_dir,
            f"cca_spectrum_{MODEL_CONFIGS[model_a]['short_name']}_{MODEL_CONFIGS[model_b]['short_name']}.png"
        )

        # 运行 CCA 分析
        pca_whitening_cca_analysis(
            X, Y,
            n_components=64,
            pca_variance_threshold=pca_variance,
            cca_corr_threshold=cca_threshold,
            spectrum_plot_path=spectrum_path,
        )

        results[(model_a, model_b)] = {
            "n_samples": X.shape[0],
            "dim_a": X.shape[1],
            "dim_b": Y.shape[1],
            "spectrum_path": spectrum_path,
        }

    return results


def run_pairwise_sae_analysis(
    model_pairs: List[Tuple[str, str]],
    cache_dir: str,
    output_dir: str,
    # SAE 配置
    shared_dim: int = 64,
    unique_x_dim: int = 16,
    unique_y_dim: int = 16,
    hidden_dim: int = 256,
    batch_size: int = 128,
    num_epochs: int = 100,
    # 损失权重
    recon_weight: float = 1.0,
    shared_recon_weight: float = 1.0,
    vicreg_weight: float = 1.0,
    ortho_weight: float = 0.1,
    unique_sparse_weight: float = 0.1,
    use_raw_mse_invariance: bool = True,
    cross_shared_recon_weight: float = 0.0,
    device: Optional[torch.device] = None,
    max_scenes: Optional[int] = None,
) -> Dict[Tuple[str, str], Dict]:
    """
    对所有模型对运行 SAE 分析

    返回: {(model_a, model_b): analysis_results}
    """
    if torch is None:
        print("警告: 未安装 PyTorch，跳过 SAE 分析")
        return {}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    print("\n" + "=" * 80)
    print("SAE 共享/特有子空间分析")
    print("=" * 80)

    for model_a, model_b in model_pairs:
        print(f"\n{'=' * 60}")
        print(f"分析: {model_a} vs {model_b}")
        print(f"{'=' * 60}")

        # 加载对齐的特征
        X, Y, common_ids, actual_a, actual_b = load_or_create_aligned_cache(
            model_a, model_b, cache_dir, max_scenes=max_scenes
        )

        # 如果顺序被交换了，需要调整
        if actual_a != model_a:
            X, Y = Y, X
            actual_a, actual_b = model_a, model_b

        # 生成输出路径
        short_a = MODEL_CONFIGS[model_a]['short_name']
        short_b = MODEL_CONFIGS[model_b]['short_name']

        model_save_path = os.path.join(output_dir, f"sae_model_{short_a}_{short_b}.pt")
        plot_path = os.path.join(output_dir, f"sae_analysis_{short_a}_{short_b}.png")

        # 运行 SAE 分析
        sae_analysis(
            X, Y,
            shared_dim=shared_dim,
            unique_x_dim=unique_x_dim,
            unique_y_dim=unique_y_dim,
            hidden_dim=hidden_dim,
            use_linear_decoder=True,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=1e-3,
            recon_weight=recon_weight,
            shared_recon_weight=shared_recon_weight,
            vicreg_weight=vicreg_weight,
            ortho_weight=ortho_weight,
            unique_sparse_weight=unique_sparse_weight,
            use_raw_mse_invariance=use_raw_mse_invariance,
            cross_shared_recon_weight=cross_shared_recon_weight,
            device=device,
            plot_path=plot_path,
            model_save_path=model_save_path,
        )

        results[(model_a, model_b)] = {
            "model_path": model_save_path,
            "plot_path": plot_path,
        }

    return results


def create_summary_report(
    cka_matrix: np.ndarray,
    model_names: List[str],
    cca_results: Dict,
    sae_results: Dict,
    output_path: str,
) -> None:
    """创建综合分析报告"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多模型表征空间分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # 模型信息
        f.write("1. 模型信息\n")
        f.write("-" * 40 + "\n")
        for name in model_names:
            config = MODEL_CONFIGS[name]
            f.write(f"  - {config['display_name']}\n")
            f.write(f"    路径: {config['path']}\n")
        f.write("\n")

        # CKA 矩阵
        f.write("2. CKA 相似度矩阵\n")
        f.write("-" * 40 + "\n")

        # 表头
        header = "          "
        for name in model_names:
            header += f"{MODEL_CONFIGS[name]['short_name']:>10}"
        f.write(header + "\n")

        # 数据行
        for i, name in enumerate(model_names):
            row = f"{MODEL_CONFIGS[name]['short_name']:<10}"
            for j in range(len(model_names)):
                row += f"{cka_matrix[i, j]:>10.4f}"
            f.write(row + "\n")
        f.write("\n")

        # CKA 排名
        f.write("3. CKA 相似度排名（从高到低）\n")
        f.write("-" * 40 + "\n")

        pairs_cka = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pairs_cka.append((model_names[i], model_names[j], cka_matrix[i, j]))

        pairs_cka.sort(key=lambda x: x[2], reverse=True)

        for rank, (a, b, cka) in enumerate(pairs_cka, 1):
            f.write(f"  {rank}. {MODEL_CONFIGS[a]['short_name']} vs {MODEL_CONFIGS[b]['short_name']}: {cka:.4f}\n")
        f.write("\n")

        # CCA 分析结果
        f.write("4. PCA + CCA 分析\n")
        f.write("-" * 40 + "\n")
        for (model_a, model_b), result in cca_results.items():
            f.write(f"  {MODEL_CONFIGS[model_a]['short_name']} vs {MODEL_CONFIGS[model_b]['short_name']}:\n")
            f.write(f"    样本数: {result['n_samples']}\n")
            f.write(f"    维度: {result['dim_a']} vs {result['dim_b']}\n")
            f.write(f"    谱图: {result['spectrum_path']}\n")
        f.write("\n")

        # SAE 分析结果
        if sae_results:
            f.write("5. SAE 共享/特有子空间分析\n")
            f.write("-" * 40 + "\n")
            for (model_a, model_b), result in sae_results.items():
                f.write(f"  {MODEL_CONFIGS[model_a]['short_name']} vs {MODEL_CONFIGS[model_b]['short_name']}:\n")
                f.write(f"    模型: {result['model_path']}\n")
                f.write(f"    图表: {result['plot_path']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("分析完成\n")
        f.write("=" * 80 + "\n")

    print(f"\n报告已保存到: {output_path}")


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description="Pairwise CKA, PCA/CCA and SAE across scene-aligned models.")
    parser.add_argument("--model", nargs=2, metavar=("NAME", "DIRECTORY"), action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sae", action="store_true")
    args = parser.parse_args()
    model_names = [name for name, _ in args.model]
    if len(set(model_names)) != len(model_names) or len(model_names) < 2:
        parser.error("Supply at least two distinct models.")
    for name, directory in args.model:
        if name not in MODEL_CONFIGS:
            parser.error(f"Unknown model {name!r}. Choose from {', '.join(MODEL_CONFIGS)}.")
        MODEL_CONFIGS[name]["path"] = directory
    if args.max_scenes is not None and args.max_scenes < 2:
        parser.error("--max-scenes must be at least 2.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_base_dir = args.output
    max_scenes = args.max_scenes

    # PCA + CCA 配置
    pca_variance = 0.99
    cca_threshold = 0.8

    # SAE 配置
    sae_config = {
        "shared_dim": 64,
        "unique_x_dim": 16,
        "unique_y_dim": 16,
        "hidden_dim": 256,
        "batch_size": 128,
        "num_epochs": args.epochs,
        "recon_weight": 1.0,
        "shared_recon_weight": 1.0,
        "vicreg_weight": 1.0,
        "ortho_weight": 0.1,
        "unique_sparse_weight": 0.1,
        "use_raw_mse_invariance": True,
        "cross_shared_recon_weight": 0.0,
    }

    # 是否运行各项分析
    run_cka = True
    run_cca = True
    run_sae = not args.skip_sae

    # 设备
    device = torch.device(args.device)

    # ============ 初始化 ============
    print("=" * 80)
    print("多模型表征空间分析")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\n分析模型: {model_names}")
    print(f"输出目录: {output_base_dir}")
    if device is not None:
        print(f"设备: {device}")

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    cache_dir = get_cache_dir(output_base_dir)

    # 生成所有模型对
    model_pairs = list(itertools.combinations(model_names, 2))
    print(f"\n共 {len(model_pairs)} 个模型对:")
    for i, (a, b) in enumerate(model_pairs, 1):
        print(f"  {i}. {a} vs {b}")

    # ============ 步骤 1: 生成对齐缓存 ============
    print("\n" + "=" * 80)
    print("步骤 1: 生成/加载对齐缓存")
    print("=" * 80)

    for model_a, model_b in model_pairs:
        load_or_create_aligned_cache(
            model_a, model_b, cache_dir, max_scenes=max_scenes, force_reload=True
        )

    # ============ 步骤 2: 计算 CKA 矩阵 ============
    cka_matrix = None
    if run_cka:
        print("\n" + "=" * 80)
        print("步骤 2: 计算 CKA 相似度矩阵")
        print("=" * 80)

        cka_matrix = compute_cka_matrix(
            model_names, cache_dir, {}, max_scenes=max_scenes
        )

        # 绘制 CKA 热图
        cka_heatmap_path = os.path.join(output_base_dir, "cka_similarity_matrix.png")
        plot_cka_heatmap(
            cka_matrix, model_names, cka_heatmap_path,
            title="Multi-Model CKA Similarity Matrix"
        )

        # 打印 CKA 矩阵
        print("\nCKA 矩阵:")
        for i, name in enumerate(model_names):
            row = f"  {MODEL_CONFIGS[name]['short_name']:<8}"
            for j in range(len(model_names)):
                row += f"{cka_matrix[i, j]:>8.4f}"
            print(row)
    else:
        # 如果不运行 CKA，创建一个空矩阵
        cka_matrix = np.eye(len(model_names))

    # ============ 步骤 3: PCA + CCA 分析 ============
    cca_results = {}
    if run_cca:
        cca_results = run_pairwise_cca_analysis(
            model_pairs, cache_dir, output_base_dir,
            pca_variance=pca_variance,
            cca_threshold=cca_threshold,
            max_scenes=max_scenes,
        )

    # ============ 步骤 4: SAE 分析 ============
    sae_results = {}
    if run_sae:
        sae_results = run_pairwise_sae_analysis(
            model_pairs, cache_dir, output_base_dir,
            device=device,
            max_scenes=max_scenes,
            **sae_config,
        )

    # ============ 步骤 5: 生成综合报告 ============
    print("\n" + "=" * 80)
    print("步骤 5: 生成综合报告")
    print("=" * 80)

    report_path = os.path.join(output_base_dir, "analysis_report.txt")
    create_summary_report(
        cka_matrix, model_names, cca_results, sae_results, report_path
    )

    # ============ 完成 ============
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 缓存目录: {cache_dir}")
    print(f"  - CKA 热图: {os.path.join(output_base_dir, 'cka_similarity_matrix.png')}")
    print(f"  - CCA 谱图: {output_base_dir}/cca_spectrum_*.png")
    print(f"  - SAE 分析: {output_base_dir}/sae_*.png, sae_*.pt")
    print(f"  - 综合报告: {report_path}")


if __name__ == "__main__":
    main()
