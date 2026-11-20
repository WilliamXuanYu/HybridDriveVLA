# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
try:
    import torch
except ImportError:
    torch = None
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_single_feature_from_obj(
    obj,
    feature_key: Optional[str] = None,
    path: Optional[str] = None,
) -> np.ndarray:
    """从 pkl 对象中提取特征向量（支持 ndarray / torch.Tensor / dict / list）"""
    # 1) 直接是 ndarray
    if isinstance(obj, np.ndarray):
        feat = obj

    # 1b) torch.Tensor
    elif torch is not None and isinstance(obj, torch.Tensor):
        feat = obj.detach().cpu().numpy()

    # 2) dict
    elif isinstance(obj, dict):
        candidate_keys: List[str] = []
        if feature_key is not None:
            candidate_keys.append(feature_key)

        candidate_keys.extend([
            "backbone_feature", "backbone_feat", "feature", "features",
            "feat", "hidden", "hidden_state", "embedding", "emb",
        ])

        for k in candidate_keys:
            if k in obj:
                feat = np.array(obj[k])
                break
        else:
            ndarray_values = [v for v in obj.values() if isinstance(v, np.ndarray)]
            if len(ndarray_values) == 1:
                feat = ndarray_values[0]
            else:
                raise ValueError(
                    f"无法在字典中自动找到特征字段，文件: {path}\n"
                    f"可用 keys: {list(obj.keys())}\n"
                    f"请手动指定 feature_key_vlm / feature_key_vit。"
                )

    # 3) list/tuple
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            feat = np.array(obj[0])
        else:
            raise ValueError(
                f"pkl 是 list/tuple 且长度 > 1，无法判断哪个是特征，文件: {path}\n"
                f"长度: {len(obj)}，请查看文件内容并修改解析逻辑。"
            )
    else:
        raise TypeError(
            f"不支持的 pkl 对象类型: {type(obj)}, 文件: {path}"
        )

    # 统一到 1D 向量
    feat = np.asarray(feat)

    if feat.ndim == 1:
        return feat.astype(np.float32)
    elif feat.ndim == 2:
        # (T, D) -> 对 T 维做平均，得到 (D,)
        feat_pooled = feat.mean(axis=0)
        return feat_pooled.astype(np.float32)
    else:
        # 高维特征：先展平除最后一维外的所有维度，再平均
        original_shape = feat.shape
        feat_reshaped = feat.reshape(-1, feat.shape[-1])  # (-1, D)
        feat_pooled = feat_reshaped.mean(axis=0)  # (D,)
        return feat_pooled.astype(np.float32)


def load_features_from_dir(
    feature_dir: str,
    feature_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """从目录加载特征（支持两种结构：直接 pkl 或 场景文件夹/pkl）"""
    features: Dict[str, np.ndarray] = {}
    scene_pkl_paths = []

    for entry in sorted(os.listdir(feature_dir)):
        entry_path = os.path.join(feature_dir, entry)

        # 情况 1: feature_dir/scene_token/*.pkl
        if os.path.isdir(entry_path):
            scene_id = entry
            pkl_files = [
                f for f in sorted(os.listdir(entry_path))
                if f.endswith(".pkl")
            ]
            if not pkl_files:
                continue

            # 优先选择包含 'backbone' 的 pkl
            chosen = None
            for fname in pkl_files:
                if "backbone" in fname.lower():
                    chosen = fname
                    break
            if chosen is None:
                chosen = pkl_files[0]

            pkl_path = os.path.join(entry_path, chosen)
            scene_pkl_paths.append((scene_id, pkl_path))

        # 情况 2: feature_dir/scene_token.pkl
        elif os.path.isfile(entry_path) and entry.endswith(".pkl"):
            scene_id = os.path.splitext(entry)[0]
            pkl_path = entry_path
            scene_pkl_paths.append((scene_id, pkl_path))

        else:
            continue

    print(f"在目录 {feature_dir} 中解析到 {len(scene_pkl_paths)} 个场景/pkl 组合。")

    for scene_id, path in tqdm(scene_pkl_paths, desc=f"Loading features from {feature_dir}"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        feat = load_single_feature_from_obj(obj, feature_key=feature_key, path=path)
        features[scene_id] = feat

    return features


def align_features(
    feats_a: Dict[str, np.ndarray],
    feats_b: Dict[str, np.ndarray],
    max_scenes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """对齐两个特征字典，返回 (X, Y, common_ids)"""
    ids_a = set(feats_a.keys())
    ids_b = set(feats_b.keys())
    common_ids = sorted(list(ids_a & ids_b))

    if len(common_ids) == 0:
        raise RuntimeError("两个目录没有共同的 scene_id，请检查命名是否一致。")

    if max_scenes is not None:
        common_ids = common_ids[:max_scenes]

    sample_a = feats_a[common_ids[0]]
    sample_b = feats_b[common_ids[0]]

    X = np.zeros((len(common_ids), sample_a.shape[0]), dtype=np.float32)
    Y = np.zeros((len(common_ids), sample_b.shape[0]), dtype=np.float32)

    for i, sid in enumerate(tqdm(common_ids, desc="对齐特征")):
        X[i] = feats_a[sid]
        Y[i] = feats_b[sid]

    print(f"对齐后的样本数: {X.shape[0]}")
    print(f"VLM feature 维度: {X.shape[1]}, ViT feature 维度: {Y.shape[1]}")

    return X, Y, common_ids


def center_features(X: np.ndarray) -> np.ndarray:
    """对特征按样本维 (axis=0) 做中心化"""
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """线性 CKA 的高效实现"""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA 要求样本数相同，得到 X: {X.shape}, Y: {Y.shape}")

    Xc = center_features(X)
    Yc = center_features(Y)

    dot = Yc.T @ Xc
    numerator = np.linalg.norm(dot, ord="fro") ** 2

    XX = Xc.T @ Xc
    YY = Yc.T @ Yc
    denom = np.linalg.norm(XX, ord="fro") * np.linalg.norm(YY, ord="fro")

    return float(numerator / denom)


def pca_whitening_cca_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 64,
    pca_variance_threshold: float = 0.99,
    cca_corr_threshold: float = 0.8,
    spectrum_plot_path: Optional[str] = None,
) -> None:
    """
    使用 PCA whitening + CCA 的严格流程进行共享/特有子空间分析。

    流程：
    1. 对 X, Y 分别做 PCA，保留 99% 方差的主成分
    2. 在 PCA 空间做 whitening（标准化每个主成分）
    3. 在 whitened PCA 空间做 CCA
    4. 根据 canonical correlation 阈值选择共享维度
    5. 提取共享/特有子空间

    参数：
    - X: VLM 特征, shape (N, Dx)
    - Y: ViT 特征, shape (N, Dy)
    - n_components: CCA 使用的最大 canonical components 数（仅用于画谱）
    - pca_variance_threshold: PCA 保留的累计方差比例
    - cca_corr_threshold: CCA canonical correlation 阈值，只保留 >= 此值的维度作为共享子空间
    - spectrum_plot_path: 谱图保存路径
    """
    n_samples = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"要求样本数相同，得到 X: {X.shape}, Y: {Y.shape}")

    print(f"\n  [Step 1] 对 X, Y 分别做 PCA（保留 {pca_variance_threshold*100:.1f}% 方差）")

    # 中心化
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # PCA on X
    pca_x = PCA(n_components=pca_variance_threshold, svd_solver='full')
    X_pca = pca_x.fit_transform(X_centered)  # (N, d_x')
    print(f"    VLM PCA: {X.shape[1]} -> {X_pca.shape[1]} 维 (保留方差: {pca_x.explained_variance_ratio_.sum():.4f})")

    # PCA on Y
    pca_y = PCA(n_components=pca_variance_threshold, svd_solver='full')
    Y_pca = pca_y.fit_transform(Y_centered)  # (N, d_y')
    print(f"    ViT PCA: {Y.shape[1]} -> {Y_pca.shape[1]} 维 (保留方差: {pca_y.explained_variance_ratio_.sum():.4f})")

    print(f"\n  [Step 2] 在 PCA 空间做 whitening（标准化主成分）")
    # Whitening: 除以每个主成分的标准差
    X_white = X_pca / np.sqrt(pca_x.explained_variance_[np.newaxis, :] + 1e-10)
    Y_white = Y_pca / np.sqrt(pca_y.explained_variance_[np.newaxis, :] + 1e-10)

    print(f"    Whitened X: mean={X_white.mean():.6f}, std={X_white.std():.6f}")
    print(f"    Whitened Y: mean={Y_white.mean():.6f}, std={Y_white.std():.6f}")

    print(f"\n  [Step 3] 在 whitened PCA 空间做 CCA")
    # CCA: 使用 SVD 求解
    # X_white: (N, dx'), Y_white: (N, dy')
    C_xy = (X_white.T @ Y_white) / n_samples  # (dx', dy')

    U, s, Vt = np.linalg.svd(C_xy, full_matrices=False)

    # canonical correlations（所有的）
    corrs = s  # shape (min(dx', dy'),)
    print(f"    SVD 得到 {len(corrs)} 个 canonical correlations")
    print(f"    前 10 个 canonical correlations: {corrs[:10]}")
    print(f"    最大值: {corrs.max():.6f}, 最小值: {corrs.min():.6f}")

    # 根据阈值选择共享维度
    keep_idx = np.where(corrs >= cca_corr_threshold)[0]
    k_shared = len(keep_idx)

    print(f"\n    使用 canonical correlation 阈值: {cca_corr_threshold}")
    print(f"    保留 {k_shared} 个维度作为共享子空间（占 {k_shared/len(corrs)*100:.2f}%）")

    if k_shared == 0:
        print(f"    警告：没有任何维度的相关性 >= {cca_corr_threshold}，共享子空间为空！")
        print(f"    建议降低阈值（当前最大相关性: {corrs.max():.6f}）")

    # CCA weights (在 whitened PCA 空间) - 只保留高相关性的维度
    W_x = U[:, keep_idx]            # (dx', k_shared)
    W_y = Vt[keep_idx, :].T         # (dy', k_shared)

    # 画谱（显示所有 canonical correlations，并标注阈值线）
    if spectrum_plot_path is not None:
        xs = np.arange(1, len(corrs) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(xs, corrs, marker="o", markersize=4, label="Canonical correlations")
        plt.axhline(y=cca_corr_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold = {cca_corr_threshold}')
        plt.xlabel("Canonical component index", fontsize=12)
        plt.ylabel("Canonical correlation", fontsize=12)
        plt.title(f"CCA canonical correlation spectrum (PCA whitening)\nShared dims: {k_shared}/{len(corrs)}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(spectrum_plot_path, dpi=200)
        plt.close()
        print(f"    谱图已保存到: {spectrum_plot_path}")

    print(f"\n  [Step 4] 提取共享/特有子空间（在 whitened PCA 空间）")

    if k_shared > 0:
        # 正交投影矩阵（whitened PCA 空间 -> shared subspace）
        # 因为 W_x, W_y 列向量已正交（SVD 保证），所以 P = W @ W^T
        Px = W_x @ W_x.T   # (dx', dx')
        Py = W_y @ W_y.T   # (dy', dy')

        # 共享子空间投影
        X_white_shared = X_white @ Px
        Y_white_shared = Y_white @ Py

        # 特有子空间
        X_white_unique = X_white - X_white_shared
        Y_white_unique = Y_white - Y_white_shared
    else:
        # 没有共享维度，全部是特有子空间
        X_white_shared = np.zeros_like(X_white)
        Y_white_shared = np.zeros_like(Y_white)
        X_white_unique = X_white.copy()
        Y_white_unique = Y_white.copy()

    # 能量统计（在 whitened 空间）
    def energy_stats_whitened(name: str, full: np.ndarray, shared: np.ndarray, unique: np.ndarray) -> None:
        full_e = np.mean(np.sum(full ** 2, axis=1))
        shared_e = np.mean(np.sum(shared ** 2, axis=1))
        unique_e = np.mean(np.sum(unique ** 2, axis=1))
        print(
            f"    [{name}] (whitened 空间) mean||full||^2={full_e:.4f}, "
            f"shared={shared_e:.4f} ({shared_e/full_e*100:.2f}%), "
            f"unique={unique_e:.4f} ({unique_e/full_e*100:.2f}%)"
        )

    print(f"  [共享/特有能量比例]（在 whitened PCA 空间）：")
    energy_stats_whitened("VLM", X_white, X_white_shared, X_white_unique)
    energy_stats_whitened("ViT", Y_white, Y_white_shared, Y_white_unique)
    # energy_stats_original("VLM", X_centered, X_white_shared, X_white_unique)
    # energy_stats_original("ViT", Y_centered, Y_white_shared, Y_white_unique)

    # 映射回原始空间（可选，用于可视化或后续分析）
    print(f"\n  [Step 5] 映射回原始特征空间")

    # whitened PCA -> PCA -> 原始空间
    # X_pca = X_white * sqrt(variance)
    X_pca_shared = X_white_shared * np.sqrt(pca_x.explained_variance_[np.newaxis, :] + 1e-10)
    Y_pca_shared = Y_white_shared * np.sqrt(pca_y.explained_variance_[np.newaxis, :] + 1e-10)

    X_pca_unique = X_white_unique * np.sqrt(pca_x.explained_variance_[np.newaxis, :] + 1e-10)
    Y_pca_unique = Y_white_unique * np.sqrt(pca_y.explained_variance_[np.newaxis, :] + 1e-10)

    # PCA -> 原始空间
    # X_shared_orig = X_pca_shared @ pca_x.components_ + X.mean(axis=0, keepdims=True)
    # Y_shared_orig = Y_pca_shared @ pca_y.components_ + Y.mean(axis=0, keepdims=True)

    # PCA -> 原始空间（全部保持为 zero-mean）
    # PCA -> 原始空间（全部保持为 zero-mean）
    X_shared_orig = X_pca_shared @ pca_x.components_  # 不加均值
    Y_shared_orig = Y_pca_shared @ pca_y.components_

    X_unique_orig = X_pca_unique @ pca_x.components_
    Y_unique_orig = Y_pca_unique @ pca_y.components_

    # 能量统计（原始空间）
    def energy_stats_original(name: str, full: np.ndarray, shared: np.ndarray, unique: np.ndarray) -> None:
        full_e = np.mean(np.sum(full ** 2, axis=1))
        shared_e = np.mean(np.sum(shared ** 2, axis=1))
        unique_e = np.mean(np.sum(unique ** 2, axis=1))
        print(
            f"    [{name}] (原始空间) mean||full||^2={full_e:.4f}, "
            f"shared={shared_e:.4f} ({shared_e/full_e*100:.2f}%), "
            f"unique={unique_e:.4f} ({unique_e/full_e*100:.2f}%)"
        )

    print(f"  [共享/特有能量比例]（映射回原始空间）：")
    # energy_stats_original("VLM", X_centered, X_unique_orig, X_unique_orig)
    # energy_stats_original("ViT", Y_centered, Y_unique_orig, Y_unique_orig)
    energy_stats_original("VLM", X_centered, X_shared_orig, X_unique_orig)
    energy_stats_original("ViT", Y_centered, Y_shared_orig, Y_unique_orig)

    print(f"\n  [Summary] CCA 分析摘要：")
    print(f"    - 总 canonical components: {len(corrs)}")
    print(f"    - 共享维度数（>= {cca_corr_threshold}）: {k_shared}")
    if k_shared > 0:
        print(f"    - 共享维度的平均 canonical correlation: {corrs[keep_idx].mean():.6f}")
        print(f"    - 所有维度的平均 canonical correlation: {corrs.mean():.6f}")
        print(f"    - 前 5 个平均: {corrs[:5].mean():.6f}")
        print(f"    - 前 10 个平均: {corrs[:10].mean():.6f}")
    else:
        print(f"    - 无共享维度（所有相关性均 < {cca_corr_threshold}）")


def main():
    import argparse
    from pair_inputs import add_pair_arguments, load_pair, parse_pair_arguments

    parser = argparse.ArgumentParser(description="Pairwise linear CKA and PCA-whitened CCA.")
    add_pair_arguments(parser)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.8])
    parser.add_argument("--pca-variance", type=float, default=0.99)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parse_pair_arguments(parser)
    X, Y, _ = load_pair(args.cache, args.x_dir, args.y_dir, args.max_scenes)
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = [("aligned", Y)]
    if args.shuffle:
        pairs.append(("shuffled", Y[np.random.RandomState(args.seed).permutation(len(Y))]))
    for label, paired_y in pairs:
        print(f"Linear CKA ({label}): {linear_cka(X, paired_y):.6f}")
        for threshold in args.threshold:
            pca_whitening_cca_analysis(
                X, paired_y, n_components=64,
                pca_variance_threshold=args.pca_variance, cca_corr_threshold=threshold,
                spectrum_plot_path=str(args.output / f"cca_spectrum_{label}_thresh{threshold:g}.png"),
            )


if __name__ == "__main__":
    main()
