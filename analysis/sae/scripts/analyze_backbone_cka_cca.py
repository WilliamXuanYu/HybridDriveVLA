# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
try:
    import torch
except ImportError:
    torch = None
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


from analyze_pca_whitening_cca import center_features, linear_cka


def compute_cca_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 64,
) -> np.ndarray:
    """
    使用 sklearn 实现 CCA，返回前 n_components 个 canonical correlation。

    常用聚合指标：
    - 均值: np.mean(corrs)
    - 前 k 个的均值，或平方和等。
    """
    n_samples = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CCA 要求样本数相同，得到 X: {X.shape}, Y: {Y.shape}")

    # 预处理：标准化每个维度，以避免尺度差异影响
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    Ys = scaler_y.fit_transform(Y)

    # CCA 的 component 数不能超过样本数 - 1，也不能超过特征维度
    max_comp = min(n_components, Xs.shape[1], Ys.shape[1], n_samples - 1)
    if max_comp <= 0:
        raise ValueError(
            f"样本数或特征维度太小，无法做 CCA，"
            f"n_samples={n_samples}, Dx={Xs.shape[1]}, Dy={Ys.shape[1]}"
        )

    print(f"\n[CCA维度变化]")
    print(f"  输入: X={X.shape}, Y={Y.shape}")
    print(f"  标准化后: Xs={Xs.shape}, Ys={Ys.shape}")
    print(f"  max_comp = min({n_components}, {Xs.shape[1]}, {Ys.shape[1]}, {n_samples - 1}) = {max_comp}")

    # 增加迭代次数和容差，提升收敛性
    cca = CCA(
        n_components=max_comp,
        max_iter=2000,      # 从 500 增加到 2000
        tol=1e-06,          # 降低收敛容差
        scale=False         # 已经手动标准化了
    )

    print(f"  CCA 配置: n_components={max_comp}, max_iter=2000, tol=1e-06")
    X_c, Y_c = cca.fit_transform(Xs, Ys)  # (N, max_comp)
    print(f"  CCA变换后: X_c={X_c.shape}, Y_c={Y_c.shape}")

    corrs = []
    for i in range(max_comp):
        x_ci = X_c[:, i]
        y_ci = Y_c[:, i]
        c = np.corrcoef(x_ci, y_ci)[0, 1]
        corrs.append(c)

    corrs_array = np.array(corrs, dtype=np.float32)
    print(f"  输出相关系数数组: {corrs_array.shape}")

    return corrs_array


def cca_shared_unique_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 64,
    spectrum_plot_path: Optional[str] = None,
) -> None:
    """基于 CCA 构造共享/特有子空间，并可选画出 canonical correlation 谱。

    - X: VLM 特征, shape (N, Dx)
    - Y: ViT 特征, shape (N, Dy)
    - n_components: 使用前 k 个 canonical components
    - spectrum_plot_path: 若不为 None，则保存谱图到该路径
    """
    n_samples = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CCA 要求样本数相同，得到 X: {X.shape}, Y: {Y.shape}")

    # 标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    Ys = scaler_y.fit_transform(Y)

    max_comp = min(n_components, Xs.shape[1], Ys.shape[1], n_samples - 1)
    if max_comp <= 0:
        raise ValueError(
            f"样本数或特征维度太小，无法做 CCA，"
            f"n_samples={n_samples}, Dx={Xs.shape[1]}, Dy={Ys.shape[1]}"
        )

    cca = CCA(
        n_components=max_comp,
        max_iter=2000,
        tol=1e-06,
        scale=False,
    )
    print(f"\n[CCA共享/特有子空间维度变化]")
    print(f"  输入: X={X.shape}, Y={Y.shape}")
    print(f"  标准化后: Xs={Xs.shape}, Ys={Ys.shape}")
    print(f"  [CCA] 共享/特有子空间分析: n_components={max_comp}, max_iter=2000, tol=1e-06")

    X_c, Y_c = cca.fit_transform(Xs, Ys)
    print(f"  CCA变换后: X_c={X_c.shape}, Y_c={Y_c.shape}")

    # canonical correlations
    corrs = []
    for i in range(max_comp):
        x_ci = X_c[:, i]
        y_ci = Y_c[:, i]
        c = np.corrcoef(x_ci, y_ci)[0, 1]
        corrs.append(c)
    corrs = np.array(corrs, dtype=np.float32)

    # 画谱
    if spectrum_plot_path is not None:
        xs = np.arange(1, max_comp + 1)
        plt.figure()
        plt.plot(xs, corrs, marker="o")
        plt.xlabel("Canonical component index")
        plt.ylabel("Correlation")
        plt.title("CCA canonical correlation spectrum")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(spectrum_plot_path, dpi=200)
        plt.close()
        print(f"  [CCA] 谱图已保存到: {spectrum_plot_path}")

    # 投影矩阵（在标准化空间）
    W_x = cca.x_weights_[:, :max_comp]   # (Dx, k)
    W_y = cca.y_weights_[:, :max_comp]   # (Dy, k)
    print(f"  权重矩阵: W_x={W_x.shape}, W_y={W_y.shape}")

    # 计算正交投影矩阵 P_x = W_x (W_x^T W_x)^{-1} W_x^T
    # 注意使用 pinv 防止数值问题
    Px = W_x @ np.linalg.pinv(W_x.T @ W_x) @ W_x.T   # (Dx, Dx)
    Py = W_y @ np.linalg.pinv(W_y.T @ W_y) @ W_y.T   # (Dy, Dy)
    print(f"  正交投影矩阵: Px={Px.shape}, Py={Py.shape}")

    # 在标准化空间中做正交投影，得到共享子空间表示
    X_shared = Xs @ Px    # (N, Dx)
    Y_shared = Ys @ Py    # (N, Dy)
    print(f"  共享子空间: X_shared={X_shared.shape}, Y_shared={Y_shared.shape}")

    # 特有部分（正交补空间）
    X_unique = Xs - X_shared
    Y_unique = Ys - Y_shared
    print(f"  特有子空间: X_unique={X_unique.shape}, Y_unique={Y_unique.shape}")

    # 打印共享/特有能量比例
    def energy_stats(name: str, full: np.ndarray, shared: np.ndarray, unique: np.ndarray) -> None:
        full_e = np.mean(np.sum(full ** 2, axis=1))
        shared_e = np.mean(np.sum(shared ** 2, axis=1))
        unique_e = np.mean(np.sum(unique ** 2, axis=1))
        print(
            f"  [{name}] mean||full||^2={full_e:.4f}, "
            f"shared={shared_e:.4f} ({shared_e/full_e*100:.2f}%), "
            f"unique={unique_e:.4f} ({unique_e/full_e*100:.2f}%)"
        )

    print("  [CCA] 共享/特有能量比例：")
    energy_stats("VLM", Xs, X_shared, X_unique)
    energy_stats("ViT", Ys, Y_shared, Y_unique)

    # 如需后续做更多分析，可在这里返回结果结构体/字典
    # 目前脚本模式直接打印即可

def main():
    import argparse
    from pair_inputs import add_pair_arguments, load_pair, parse_pair_arguments

    parser = argparse.ArgumentParser(description="Pairwise linear CKA and direct CCA.")
    add_pair_arguments(parser)
    parser.add_argument("--components", type=int, default=64)
    args = parse_pair_arguments(parser)
    X, Y, _ = load_pair(args.cache, args.x_dir, args.y_dir, args.max_scenes)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Linear CKA: {linear_cka(X, Y):.6f}")
    cca_shared_unique_analysis(
        X, Y, n_components=args.components,
        spectrum_plot_path=str(args.output / "cca_spectrum.png"),
    )


if __name__ == "__main__":
    main()
