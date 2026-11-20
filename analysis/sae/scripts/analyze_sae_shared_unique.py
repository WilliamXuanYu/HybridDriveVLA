# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
import os
import pickle
from typing import Dict, List, Tuple, Optional

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
import pandas as pd

# 复用原有代码中的函数
from analyze_pca_whitening_cca import (
    load_single_feature_from_obj,
    load_features_from_dir,
    align_features,
    center_features,
    linear_cka,
)


class SharedUniqueSAE(nn.Module):
    """
    改进版 SAE：学习共享/特有子空间

    关键改进：
    1. 线性解码器（可加性，确保 shared + unique 可解释）
    2. VICReg 风格的防塌缩机制
    3. Batch 协方差正交约束
    4. Shared 单独重构约束
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        shared_dim: int = 64,
        unique_x_dim: int = 16,  # 默认减小 unique 容量
        unique_y_dim: int = 16,
        hidden_dim: int = 256,
        use_linear_decoder: bool = True,  # 使用线性解码器以保证可加性
    ):
        super(SharedUniqueSAE, self).__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.shared_dim = shared_dim
        self.unique_x_dim = unique_x_dim
        self.unique_y_dim = unique_y_dim
        self.use_linear_decoder = use_linear_decoder

        # 共享编码器（非线性）
        self.shared_encoder_x = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim),
        )

        self.shared_encoder_y = nn.Sequential(
            nn.Linear(dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim),
        )

        # 特有编码器（非线性）
        self.unique_encoder_x = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, unique_x_dim),
        )

        self.unique_encoder_y = nn.Sequential(
            nn.Linear(dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, unique_y_dim),
        )

        # 解码器：使用线性可加形式（保证 shared + unique 可解释）
        # x_recon = W_s @ shared + W_u @ unique + bias
        if use_linear_decoder:
            # 线性解码器
            self.decoder_x_shared = nn.Linear(shared_dim, dim_x, bias=False)
            self.decoder_x_unique = nn.Linear(unique_x_dim, dim_x, bias=False)
            self.decoder_x_bias = nn.Parameter(torch.zeros(dim_x))

            self.decoder_y_shared = nn.Linear(shared_dim, dim_y, bias=False)
            self.decoder_y_unique = nn.Linear(unique_y_dim, dim_y, bias=False)
            self.decoder_y_bias = nn.Parameter(torch.zeros(dim_y))
        else:
            # 非线性解码器（fallback）
            self.decoder_x = nn.Sequential(
                nn.Linear(shared_dim + unique_x_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim_x),
            )

            self.decoder_y = nn.Sequential(
                nn.Linear(shared_dim + unique_y_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim_y),
            )

    def decode_x(self, shared_x, unique_x):
        """解码 X"""
        if self.use_linear_decoder:
            # 线性可加: x = W_s @ s + W_u @ u + b
            return self.decoder_x_shared(shared_x) + self.decoder_x_unique(unique_x) + self.decoder_x_bias
        else:
            return self.decoder_x(torch.cat([shared_x, unique_x], dim=1))

    def decode_y(self, shared_y, unique_y):
        """解码 Y"""
        if self.use_linear_decoder:
            return self.decoder_y_shared(shared_y) + self.decoder_y_unique(unique_y) + self.decoder_y_bias
        else:
            return self.decoder_y(torch.cat([shared_y, unique_y], dim=1))

    def decode_x_shared_only(self, shared_x):
        """仅用 shared 解码 X（用于评估）"""
        if self.use_linear_decoder:
            return self.decoder_x_shared(shared_x) + self.decoder_x_bias
        else:
            zero_unique = torch.zeros(shared_x.shape[0], self.unique_x_dim, device=shared_x.device)
            return self.decoder_x(torch.cat([shared_x, zero_unique], dim=1))

    def decode_y_shared_only(self, shared_y):
        """仅用 shared 解码 Y（用于评估）"""
        if self.use_linear_decoder:
            return self.decoder_y_shared(shared_y) + self.decoder_y_bias
        else:
            zero_unique = torch.zeros(shared_y.shape[0], self.unique_y_dim, device=shared_y.device)
            return self.decoder_y(torch.cat([shared_y, zero_unique], dim=1))

    def decode_x_unique_only(self, unique_x):
        """仅用 unique 解码 X（用于评估）"""
        if self.use_linear_decoder:
            return self.decoder_x_unique(unique_x) + self.decoder_x_bias
        else:
            zero_shared = torch.zeros(unique_x.shape[0], self.shared_dim, device=unique_x.device)
            return self.decoder_x(torch.cat([zero_shared, unique_x], dim=1))

    def decode_y_unique_only(self, unique_y):
        """仅用 unique 解码 Y（用于评估）"""
        if self.use_linear_decoder:
            return self.decoder_y_unique(unique_y) + self.decoder_y_bias
        else:
            zero_shared = torch.zeros(unique_y.shape[0], self.shared_dim, device=unique_y.device)
            return self.decoder_y(torch.cat([zero_shared, unique_y], dim=1))

    def forward(self, x, y):
        """前向传播"""
        # 编码
        shared_x = self.shared_encoder_x(x)
        shared_y = self.shared_encoder_y(y)
        unique_x = self.unique_encoder_x(x)
        unique_y = self.unique_encoder_y(y)

        # 解码
        x_recon = self.decode_x(shared_x, unique_x)
        y_recon = self.decode_y(shared_y, unique_y)

        # 仅 shared 解码（用于 shared 重构约束）
        x_shared_recon = self.decode_x_shared_only(shared_x)
        y_shared_recon = self.decode_y_shared_only(shared_y)

        return {
            'shared_x': shared_x,
            'shared_y': shared_y,
            'unique_x': unique_x,
            'unique_y': unique_y,
            'x_recon': x_recon,
            'y_recon': y_recon,
            'x_shared_recon': x_shared_recon,
            'y_shared_recon': y_shared_recon,
        }


def compute_vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0,
                        use_raw_mse_invariance=False):
    """
    VICReg 风格的损失函数（防止表示塌缩）

    参数：
    - z1, z2: 两个表示，应该相似
    - sim_weight: 相似性（invariance）权重
    - var_weight: 方差权重（防塌缩）
    - cov_weight: 协方差权重（去冗余）
    - use_raw_mse_invariance: 是否使用原版 MSE（不做标准化），默认 False

    返回损失字典
    """
    batch_size, dim = z1.shape

    # 1. Invariance: 鼓励 z1 和 z2 相似
    if use_raw_mse_invariance:
        # 原版：直接 MSE(z1, z2)，不做标准化
        inv_loss = nn.functional.mse_loss(z1, z2)
    else:
        # VICReg 变体：先 normalize 再算 MSE
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        inv_loss = nn.functional.mse_loss(z1_norm, z2_norm)

    # 2. Variance: 鼓励每一维有足够方差（std > 1，防止输出全 0）
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-6)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-6)
    # hinge loss: max(0, 1 - std)
    var_loss = torch.mean(torch.relu(1 - std_z1)) + torch.mean(torch.relu(1 - std_z2))

    # 3. Covariance: 去冗余（让不同维度去相关）
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
    # 只惩罚非对角元素
    cov_loss = (off_diagonal(cov_z1).pow(2).sum() / dim +
                off_diagonal(cov_z2).pow(2).sum() / dim)

    total = sim_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss

    return {
        'vicreg_total': total,
        'vicreg_inv': inv_loss,
        'vicreg_var': var_loss,
        'vicreg_cov': cov_loss,
    }


def off_diagonal(x):
    """提取矩阵的非对角元素"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_orthogonality_loss_batch_cov(shared, unique):
    """
    Batch 协方差版本的正交性约束

    确保 shared 和 unique 在统计意义上正交（相关性为 0）
    使用 N-1 作为分母（更标准的无偏估计）
    """
    # 中心化
    S = shared - shared.mean(dim=0, keepdim=True)
    U = unique - unique.mean(dim=0, keepdim=True)

    batch_size = S.shape[0]

    # 计算交叉协方差矩阵（使用 N-1 作为分母）
    # S: (B, d_s), U: (B, d_u)
    # cross_cov: (d_s, d_u)
    cross_cov = S.T @ U / (batch_size - 1)

    # Frobenius 范数的平方
    ortho_loss = torch.norm(cross_cov, p='fro') ** 2

    return ortho_loss


def compute_loss(
    model: SharedUniqueSAE,
    x: torch.Tensor,
    y: torch.Tensor,
    outputs: Dict,
    # 权重配置
    recon_weight: float = 1.0,
    shared_recon_weight: float = 1.0,  # 共享单独重构权重
    vicreg_weight: float = 1.0,        # VICReg 权重
    ortho_weight: float = 0.1,         # 正交约束权重
    unique_sparse_weight: float = 0.1, # unique 稀疏惩罚（只对 unique 加）
    # 新增选项
    use_raw_mse_invariance: bool = False,  # 是否使用原版 MSE 作为 invariance
    cross_shared_recon_weight: float = 0.0,  # 跨模态 shared 重构权重
) -> Dict:
    """
    改进的损失函数

    损失组成：
    1. 重构损失：完整重构
    2. 共享重构损失：仅用 shared 重构，鼓励 shared 先吃掉共同结构
    3. VICReg 损失：防止 shared 塌缩
    4. 正交约束：batch 协方差版本
    5. Unique 稀疏惩罚：让 unique 只补残差
    6. [可选] 跨模态 shared 重构：强约束 shared 可互换
    """
    shared_x = outputs['shared_x']
    shared_y = outputs['shared_y']
    unique_x = outputs['unique_x']
    unique_y = outputs['unique_y']
    x_recon = outputs['x_recon']
    y_recon = outputs['y_recon']
    x_shared_recon = outputs['x_shared_recon']
    y_shared_recon = outputs['y_shared_recon']

    # 1. 完整重构损失
    recon_loss_x = nn.functional.mse_loss(x_recon, x)
    recon_loss_y = nn.functional.mse_loss(y_recon, y)
    recon_loss = recon_loss_x + recon_loss_y

    # 2. 共享单独重构损失（鼓励 shared 先把能解释的共同结构吃掉）
    shared_recon_loss_x = nn.functional.mse_loss(x_shared_recon, x)
    shared_recon_loss_y = nn.functional.mse_loss(y_shared_recon, y)
    shared_recon_loss = shared_recon_loss_x + shared_recon_loss_y

    # 3. VICReg 损失（防止 shared 塌缩，鼓励 shared_x 和 shared_y 对齐）
    vicreg_losses = compute_vicreg_loss(
        shared_x, shared_y,
        sim_weight=25.0,  # 相似性
        var_weight=25.0,  # 方差
        cov_weight=1.0,   # 去冗余
        use_raw_mse_invariance=use_raw_mse_invariance,  # 是否使用原版 MSE
    )
    vicreg_loss = vicreg_losses['vicreg_total']

    # 4. 正交约束（batch 协方差版本）
    ortho_loss_x = compute_orthogonality_loss_batch_cov(shared_x, unique_x)
    ortho_loss_y = compute_orthogonality_loss_batch_cov(shared_y, unique_y)
    ortho_loss = ortho_loss_x + ortho_loss_y

    # 5. Unique 稀疏惩罚（只对 unique 加 L1，shared 不加）
    unique_sparse_loss = (
        torch.mean(torch.abs(unique_x)) +
        torch.mean(torch.abs(unique_y))
    )

    # 6. 跨模态 shared 重构损失（强约束 shared 可互换）
    cross_shared_recon_loss = torch.tensor(0.0, device=x.device)
    if cross_shared_recon_weight > 0:
        # 用 shared_y 解码 X
        x_from_shared_y = model.decode_x_shared_only(shared_y)
        cross_recon_loss_x = nn.functional.mse_loss(x_from_shared_y, x)
        # 用 shared_x 解码 Y
        y_from_shared_x = model.decode_y_shared_only(shared_x)
        cross_recon_loss_y = nn.functional.mse_loss(y_from_shared_x, y)
        cross_shared_recon_loss = cross_recon_loss_x + cross_recon_loss_y

    # 总损失
    total_loss = (
        recon_weight * recon_loss +
        shared_recon_weight * shared_recon_loss +
        vicreg_weight * vicreg_loss +
        ortho_weight * ortho_loss +
        unique_sparse_weight * unique_sparse_loss +
        cross_shared_recon_weight * cross_shared_recon_loss
    )

    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'shared_recon_loss': shared_recon_loss,
        'vicreg_loss': vicreg_loss,
        'vicreg_inv': vicreg_losses['vicreg_inv'],
        'vicreg_var': vicreg_losses['vicreg_var'],
        'vicreg_cov': vicreg_losses['vicreg_cov'],
        'ortho_loss': ortho_loss,
        'unique_sparse_loss': unique_sparse_loss,
        'cross_shared_recon_loss': cross_shared_recon_loss,
    }


def train_sae_model(
    X: np.ndarray,
    Y: np.ndarray,
    shared_dim: int = 64,
    unique_x_dim: int = 16,  # 减小 unique 容量
    unique_y_dim: int = 16,
    hidden_dim: int = 256,
    use_linear_decoder: bool = True,
    batch_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    # 损失权重
    recon_weight: float = 1.0,
    shared_recon_weight: float = 1.0,
    vicreg_weight: float = 1.0,
    ortho_weight: float = 0.1,
    unique_sparse_weight: float = 0.1,
    # 新增选项
    use_raw_mse_invariance: bool = False,  # 是否使用原版 MSE 作为 invariance
    cross_shared_recon_weight: float = 0.0,  # 跨模态 shared 重构权重
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[SharedUniqueSAE, Dict]:
    """训练 SAE 模型学习共享/特有子空间"""
    if torch is None:
        raise ImportError("需要安装 PyTorch 才能使用 SAE 训练")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples, dim_x = X.shape
    _, dim_y = Y.shape

    # 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X).astype(np.float32)
    Y_scaled = scaler_y.fit_transform(Y).astype(np.float32)

    # 转换为 PyTorch 张量
    X_tensor = torch.from_numpy(X_scaled).to(device)
    Y_tensor = torch.from_numpy(Y_scaled).to(device)

    # 创建数据加载器
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) % batch_size == 1
    )

    # 创建模型
    model = SharedUniqueSAE(
        dim_x=dim_x,
        dim_y=dim_y,
        shared_dim=shared_dim,
        unique_x_dim=unique_x_dim,
        unique_y_dim=unique_y_dim,
        hidden_dim=hidden_dim,
        use_linear_decoder=use_linear_decoder,
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练历史
    history = {
        'total_loss': [],
        'recon_loss': [],
        'shared_recon_loss': [],
        'vicreg_loss': [],
        'vicreg_var': [],
        'ortho_loss': [],
        'unique_sparse_loss': [],
        'cross_shared_recon_loss': [],
    }

    if verbose:
        print(f"\n开始训练 SAE 模型...")
        print(f"  设备: {device}")
        print(f"  样本数: {n_samples}")
        print(f"  VLM 维度: {dim_x}, ViT 维度: {dim_y}")
        print(f"  共享维度: {shared_dim}, X特有维度: {unique_x_dim}, Y特有维度: {unique_y_dim}")
        print(f"  解码器类型: {'线性' if use_linear_decoder else '非线性'}")
        print(f"  批次大小: {batch_size}, 训练轮数: {num_epochs}")
        print(f"  损失权重: recon={recon_weight}, shared_recon={shared_recon_weight}, "
              f"vicreg={vicreg_weight}, ortho={ortho_weight}, unique_sparse={unique_sparse_weight}")
        print(f"  新增选项: use_raw_mse_invariance={use_raw_mse_invariance}, "
              f"cross_shared_recon_weight={cross_shared_recon_weight}")

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = {key: 0.0 for key in history.keys()}

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_x, batch_y)
            losses = compute_loss(
                model, batch_x, batch_y, outputs,
                recon_weight=recon_weight,
                shared_recon_weight=shared_recon_weight,
                vicreg_weight=vicreg_weight,
                ortho_weight=ortho_weight,
                unique_sparse_weight=unique_sparse_weight,
                use_raw_mse_invariance=use_raw_mse_invariance,
                cross_shared_recon_weight=cross_shared_recon_weight,
            )

            # 反向传播
            losses['total_loss'].backward()
            optimizer.step()

            # 累计损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

        # 平均损失
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])

        # 打印进度
        if verbose and (epoch + 1) % 10 == 0:
            msg = (
                f"  Epoch {epoch+1}/{num_epochs}: "
                f"total={epoch_losses['total_loss']:.4f}, "
                f"recon={epoch_losses['recon_loss']:.4f}, "
                f"shared_recon={epoch_losses['shared_recon_loss']:.4f}, "
                f"vicreg={epoch_losses['vicreg_loss']:.4f}, "
                f"vicreg_var={epoch_losses['vicreg_var']:.4f}, "
                f"ortho={epoch_losses['ortho_loss']:.4f}"
            )
            if cross_shared_recon_weight > 0:
                msg += f", cross_recon={epoch_losses['cross_shared_recon_loss']:.4f}"
            print(msg)

    if verbose:
        print("训练完成！")

    # 保存 scaler 到模型
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y

    return model, history


def sae_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    shared_dim: int = 64,
    unique_x_dim: int = 16,
    unique_y_dim: int = 16,
    hidden_dim: int = 256,
    use_linear_decoder: bool = True,
    batch_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    recon_weight: float = 1.0,
    shared_recon_weight: float = 1.0,
    vicreg_weight: float = 1.0,
    ortho_weight: float = 0.1,
    unique_sparse_weight: float = 0.1,
    # 新增选项
    use_raw_mse_invariance: bool = False,  # 是否使用原版 MSE 作为 invariance
    cross_shared_recon_weight: float = 0.0,  # 跨模态 shared 重构权重
    device: Optional[torch.device] = None,
    plot_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    return_metrics: bool = False,  # 是否返回指标（用于批量实验）
) -> Optional[Dict]:
    """
    使用 SAE 进行共享/特有子空间分析

    改进：
    1. 所有评估在 scaled 空间进行
    2. 使用 ablation 方式评估贡献（而非能量分解）
    3. 正确的共享空间对齐检验

    新增选项：
    - use_raw_mse_invariance: 使用原版 MSE(shared_x, shared_y) 代替标准化后的 MSE
    - cross_shared_recon_weight: 跨模态 shared 重构权重，>0 时启用
    - return_metrics: 是否返回关键指标字典（用于批量实验汇总）
    """
    n_samples = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"要求样本数相同，得到 X: {X.shape}, Y: {Y.shape}")

    print(f"\n{'='*80}")
    print(f"SAE 共享/特有子空间分析（改进版）")
    print(f"{'='*80}")
    print(f"  样本数: {n_samples}")
    print(f"  VLM 特征维度: {X.shape[1]}")
    print(f"  ViT 特征维度: {Y.shape[1]}")
    print(f"  共享子空间维度: {shared_dim}")
    print(f"  X 特有子空间维度: {unique_x_dim}")
    print(f"  Y 特有子空间维度: {unique_y_dim}")
    print(f"  解码器类型: {'线性（可加）' if use_linear_decoder else '非线性'}")
    print(f"  [实验选项] use_raw_mse_invariance={use_raw_mse_invariance}, "
          f"cross_shared_recon_weight={cross_shared_recon_weight}")

    # 训练模型
    print(f"\n  [Step 1] 训练 SAE 模型...")
    model, history = train_sae_model(
        X, Y,
        shared_dim=shared_dim,
        unique_x_dim=unique_x_dim,
        unique_y_dim=unique_y_dim,
        hidden_dim=hidden_dim,
        use_linear_decoder=use_linear_decoder,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        recon_weight=recon_weight,
        shared_recon_weight=shared_recon_weight,
        vicreg_weight=vicreg_weight,
        ortho_weight=ortho_weight,
        unique_sparse_weight=unique_sparse_weight,
        use_raw_mse_invariance=use_raw_mse_invariance,
        cross_shared_recon_weight=cross_shared_recon_weight,
        device=device,
        verbose=True,
    )

    # 保存模型
    if model_save_path is not None:
        print(f"\n  [Step 2] 保存模型到: {model_save_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_x': model.scaler_x,
            'scaler_y': model.scaler_y,
            'config': {
                'dim_x': X.shape[1],
                'dim_y': Y.shape[1],
                'shared_dim': shared_dim,
                'unique_x_dim': unique_x_dim,
                'unique_y_dim': unique_y_dim,
                'hidden_dim': hidden_dim,
                'use_linear_decoder': use_linear_decoder,
            }
        }, model_save_path)
        print(f"  模型已保存")

    # ============ 评估（全部在 scaled 空间进行）============
    print(f"\n  [Step 3] 在 scaled 空间进行评估...")

    # 标准化数据
    X_scaled = model.scaler_x.transform(X).astype(np.float32)
    Y_scaled = model.scaler_y.transform(Y).astype(np.float32)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        Y_tensor = torch.from_numpy(Y_scaled).to(device)

        # 编码
        shared_x = model.shared_encoder_x(X_tensor)
        shared_y = model.shared_encoder_y(Y_tensor)
        unique_x = model.unique_encoder_x(X_tensor)
        unique_y = model.unique_encoder_y(Y_tensor)

        # 完整重构
        x_recon = model.decode_x(shared_x, unique_x)
        y_recon = model.decode_y(shared_y, unique_y)

        # 仅 shared 重构
        x_shared_recon = model.decode_x_shared_only(shared_x)
        y_shared_recon = model.decode_y_shared_only(shared_y)

        # 仅 unique 重构
        x_unique_recon = model.decode_x_unique_only(unique_x)
        y_unique_recon = model.decode_y_unique_only(unique_y)

        # 转为 numpy
        shared_x_np = shared_x.cpu().numpy()
        shared_y_np = shared_y.cpu().numpy()
        unique_x_np = unique_x.cpu().numpy()
        unique_y_np = unique_y.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()
        y_recon_np = y_recon.cpu().numpy()
        x_shared_recon_np = x_shared_recon.cpu().numpy()
        y_shared_recon_np = y_shared_recon.cpu().numpy()
        x_unique_recon_np = x_unique_recon.cpu().numpy()
        y_unique_recon_np = y_unique_recon.cpu().numpy()

    # 3.1 重构误差（scaled 空间）
    print(f"\n  [3.1] 重构误差 (scaled 空间, MSE):")
    recon_mse_x = np.mean((X_scaled - x_recon_np) ** 2)
    recon_mse_y = np.mean((Y_scaled - y_recon_np) ** 2)
    print(f"    X 完整重构 MSE: {recon_mse_x:.6f}")
    print(f"    Y 完整重构 MSE: {recon_mse_y:.6f}")

    shared_recon_mse_x = np.mean((X_scaled - x_shared_recon_np) ** 2)
    shared_recon_mse_y = np.mean((Y_scaled - y_shared_recon_np) ** 2)
    print(f"    X 仅shared重构 MSE: {shared_recon_mse_x:.6f}")
    print(f"    Y 仅shared重构 MSE: {shared_recon_mse_y:.6f}")

    # 3.2 方差解释比例（ablation 方式，不是能量分解）
    print(f"\n  [3.2] 方差解释比例 (ablation, scaled 空间):")
    # 原始方差（使用 ddof=1 无偏估计，与后续方差分解一致）
    var_x = np.var(X_scaled, ddof=1)
    var_y = np.var(Y_scaled, ddof=1)

    # R² = 1 - MSE / Var
    r2_full_x = 1 - recon_mse_x / var_x
    r2_full_y = 1 - recon_mse_y / var_y
    r2_shared_x = 1 - shared_recon_mse_x / var_x
    r2_shared_y = 1 - shared_recon_mse_y / var_y

    print(f"    X 完整重构 R²: {r2_full_x:.4f} ({r2_full_x*100:.2f}%)")
    print(f"    Y 完整重构 R²: {r2_full_y:.4f} ({r2_full_y*100:.2f}%)")
    print(f"    X 仅shared R²: {r2_shared_x:.4f} ({r2_shared_x*100:.2f}%)")
    print(f"    Y 仅shared R²: {r2_shared_y:.4f} ({r2_shared_y*100:.2f}%)")

    # 3.3 编码空间统计
    print(f"\n  [3.3] 编码空间统计:")

    # Shared 方差（VICReg 目标：每维 std > 1）
    shared_x_std = np.std(shared_x_np, axis=0)
    shared_y_std = np.std(shared_y_np, axis=0)
    print(f"    Shared_x 各维std: mean={shared_x_std.mean():.4f}, min={shared_x_std.min():.4f}, max={shared_x_std.max():.4f}")
    print(f"    Shared_y 各维std: mean={shared_y_std.mean():.4f}, min={shared_y_std.min():.4f}, max={shared_y_std.max():.4f}")

    # Unique 稀疏性
    unique_x_l1 = np.mean(np.abs(unique_x_np))
    unique_y_l1 = np.mean(np.abs(unique_y_np))
    print(f"    Unique_x 平均 L1: {unique_x_l1:.6f}")
    print(f"    Unique_y 平均 L1: {unique_y_l1:.6f}")

    # 3.4 Shared 空间对齐（VICReg invariance）
    print(f"\n  [3.4] Shared 空间对齐:")

    # 标准化后的 MSE
    shared_x_norm = (shared_x_np - shared_x_np.mean(axis=0)) / (shared_x_np.std(axis=0) + 1e-6)
    shared_y_norm = (shared_y_np - shared_y_np.mean(axis=0)) / (shared_y_np.std(axis=0) + 1e-6)
    shared_align_mse = np.mean((shared_x_norm - shared_y_norm) ** 2)
    print(f"    Shared 对齐 MSE (normalized): {shared_align_mse:.6f}")

    # 逐样本余弦相似度
    cosine_sims = []
    for i in range(n_samples):
        norm_x = np.linalg.norm(shared_x_np[i])
        norm_y = np.linalg.norm(shared_y_np[i])
        if norm_x > 1e-8 and norm_y > 1e-8:
            cosine_sims.append(np.dot(shared_x_np[i], shared_y_np[i]) / (norm_x * norm_y))
    cosine_sim_mean = np.mean(cosine_sims) if cosine_sims else 0
    print(f"    Shared 平均余弦相似度: {cosine_sim_mean:.6f}")

    # CKA
    cka_shared = linear_cka(shared_x_np, shared_y_np)
    print(f"    Shared 空间 CKA: {cka_shared:.6f}")

    # 3.5 正交性检验（batch 协方差，使用 N-1）
    print(f"\n  [3.5] Shared-Unique 正交性 (latent 空间):")

    # 计算交叉协方差矩阵的 Frobenius 范数（使用 N-1）
    S_x = shared_x_np - shared_x_np.mean(axis=0)
    U_x = unique_x_np - unique_x_np.mean(axis=0)
    cross_cov_x = S_x.T @ U_x / (n_samples - 1)
    ortho_x = np.linalg.norm(cross_cov_x, 'fro') ** 2

    S_y = shared_y_np - shared_y_np.mean(axis=0)
    U_y = unique_y_np - unique_y_np.mean(axis=0)
    cross_cov_y = S_y.T @ U_y / (n_samples - 1)
    ortho_y = np.linalg.norm(cross_cov_y, 'fro') ** 2

    print(f"    X: ||Cov(Shared, Unique)||_F² = {ortho_x:.6f}")
    print(f"    Y: ||Cov(Shared, Unique)||_F² = {ortho_y:.6f}")

    # 3.6 原始空间 CKA 对比
    print(f"\n  [3.6] CKA 相似度对比:")
    cka_original = linear_cka(X, Y)
    cka_shared = linear_cka(shared_x_np, shared_y_np)

    print(f"    原始特征 CKA: {cka_original:.6f}")
    print(f"    Shared 空间 CKA: {cka_shared:.6f}")

    if unique_x_dim == unique_y_dim:
        cka_unique = linear_cka(unique_x_np, unique_y_np)
        print(f"    Unique 空间 CKA: {cka_unique:.6f}")
    else:
        cka_unique = None
        print(f"    Unique 空间 CKA: N/A (维度不同)")

    # ============ 新增评估 ============

    # 3.7 跨模态解码/交叉重构评估
    print(f"\n  [3.7] 跨模态交叉重构评估 (验证 shared 是否真正跨模态可用):")

    with torch.no_grad():
        shared_x_tensor = torch.from_numpy(shared_x_np.astype(np.float32)).to(device)
        shared_y_tensor = torch.from_numpy(shared_y_np.astype(np.float32)).to(device)
        unique_x_tensor = torch.from_numpy(unique_x_np.astype(np.float32)).to(device)
        unique_y_tensor = torch.from_numpy(unique_y_np.astype(np.float32)).to(device)

        # 用 shared_y 解码 X（完全不用 X 的信息，只用 Y 的 shared）
        x_from_shared_y_only = model.decode_x_shared_only(shared_y_tensor).cpu().numpy()
        # 用 shared_x 解码 Y
        y_from_shared_x_only = model.decode_y_shared_only(shared_x_tensor).cpu().numpy()

        # 混合：用 shared_y + unique_x 解码 X
        x_from_shared_y_unique_x = model.decode_x(shared_y_tensor, unique_x_tensor).cpu().numpy()
        # 混合：用 shared_x + unique_y 解码 Y
        y_from_shared_x_unique_y = model.decode_y(shared_x_tensor, unique_y_tensor).cpu().numpy()

    # 计算交叉重构误差
    cross_mse_x_from_shared_y = np.mean((X_scaled - x_from_shared_y_only) ** 2)
    cross_mse_y_from_shared_x = np.mean((Y_scaled - y_from_shared_x_only) ** 2)
    cross_mse_x_mixed = np.mean((X_scaled - x_from_shared_y_unique_x) ** 2)
    cross_mse_y_mixed = np.mean((Y_scaled - y_from_shared_x_unique_y) ** 2)

    # 计算 R²
    r2_x_from_shared_y = 1 - cross_mse_x_from_shared_y / var_x
    r2_y_from_shared_x = 1 - cross_mse_y_from_shared_x / var_y
    r2_x_mixed = 1 - cross_mse_x_mixed / var_x
    r2_y_mixed = 1 - cross_mse_y_mixed / var_y

    print(f"    [仅 shared 跨模态]:")
    print(f"      X <- shared_y only: MSE={cross_mse_x_from_shared_y:.6f}, R²={r2_x_from_shared_y:.4f} ({r2_x_from_shared_y*100:.2f}%)")
    print(f"      Y <- shared_x only: MSE={cross_mse_y_from_shared_x:.6f}, R²={r2_y_from_shared_x:.4f} ({r2_y_from_shared_x*100:.2f}%)")
    print(f"    [混合 shared 跨模态 + 自身 unique]:")
    print(f"      X <- shared_y + unique_x: MSE={cross_mse_x_mixed:.6f}, R²={r2_x_mixed:.4f} ({r2_x_mixed*100:.2f}%)")
    print(f"      Y <- shared_x + unique_y: MSE={cross_mse_y_mixed:.6f}, R²={r2_y_mixed:.4f} ({r2_y_mixed*100:.2f}%)")

    # 对比：自身 shared vs 跨模态 shared
    print(f"    [对比] 自身 shared vs 跨模态 shared:")
    print(f"      X: 自身 R²={r2_shared_x:.4f} vs 跨模态 R²={r2_x_from_shared_y:.4f}, 差异={r2_shared_x - r2_x_from_shared_y:.4f}")
    print(f"      Y: 自身 R²={r2_shared_y:.4f} vs 跨模态 R²={r2_y_from_shared_x:.4f}, 差异={r2_shared_y - r2_y_from_shared_x:.4f}")

    # 3.8 解码器权重诊断（仅线性解码器）
    if use_linear_decoder:
        print(f"\n  [3.8] 解码器权重诊断 (线性解码器):")

        W_s_x = model.decoder_x_shared.weight.detach().cpu().numpy()  # (dim_x, shared_dim)
        W_u_x = model.decoder_x_unique.weight.detach().cpu().numpy()  # (dim_x, unique_x_dim)
        W_s_y = model.decoder_y_shared.weight.detach().cpu().numpy()  # (dim_y, shared_dim)
        W_u_y = model.decoder_y_unique.weight.detach().cpu().numpy()  # (dim_y, unique_y_dim)

        # Frobenius 范数
        norm_W_s_x = np.linalg.norm(W_s_x, 'fro')
        norm_W_u_x = np.linalg.norm(W_u_x, 'fro')
        norm_W_s_y = np.linalg.norm(W_s_y, 'fro')
        norm_W_u_y = np.linalg.norm(W_u_y, 'fro')

        print(f"    权重 Frobenius 范数:")
        print(f"      X: ||W_s||_F={norm_W_s_x:.4f}, ||W_u||_F={norm_W_u_x:.4f}, 比值={norm_W_u_x/norm_W_s_x:.4f}")
        print(f"      Y: ||W_s||_F={norm_W_s_y:.4f}, ||W_u||_F={norm_W_u_y:.4f}, 比值={norm_W_u_y/norm_W_s_y:.4f}")

        # 每维平均权重范数
        avg_col_norm_W_s_x = np.mean(np.linalg.norm(W_s_x, axis=0))
        avg_col_norm_W_u_x = np.mean(np.linalg.norm(W_u_x, axis=0))
        avg_col_norm_W_s_y = np.mean(np.linalg.norm(W_s_y, axis=0))
        avg_col_norm_W_u_y = np.mean(np.linalg.norm(W_u_y, axis=0))

        print(f"    每编码维度平均权重范数:")
        print(f"      X: W_s 每列平均={avg_col_norm_W_s_x:.4f}, W_u 每列平均={avg_col_norm_W_u_x:.4f}")
        print(f"      Y: W_s 每列平均={avg_col_norm_W_s_y:.4f}, W_u 每列平均={avg_col_norm_W_u_y:.4f}")

        # Unique 编码的方差（补充 3.3 中的 L1）
        unique_x_std = np.std(unique_x_np, axis=0)
        unique_y_std = np.std(unique_y_np, axis=0)
        print(f"    Unique 编码统计:")
        print(f"      Unique_x std: mean={unique_x_std.mean():.6f}, min={unique_x_std.min():.6f}, max={unique_x_std.max():.6f}")
        print(f"      Unique_y std: mean={unique_y_std.mean():.6f}, min={unique_y_std.min():.6f}, max={unique_y_std.max():.6f}")

        # 有效贡献 = 权重范数 × 编码 std
        effective_contrib_s_x = avg_col_norm_W_s_x * shared_x_std.mean()
        effective_contrib_u_x = avg_col_norm_W_u_x * unique_x_std.mean()
        effective_contrib_s_y = avg_col_norm_W_s_y * shared_y_std.mean()
        effective_contrib_u_y = avg_col_norm_W_u_y * unique_y_std.mean()

        print(f"    有效贡献 (权重范数 × 编码std):")
        print(f"      X: shared={effective_contrib_s_x:.4f}, unique={effective_contrib_u_x:.4f}, 比值={effective_contrib_u_x/effective_contrib_s_x:.4f}")
        print(f"      Y: shared={effective_contrib_s_y:.4f}, unique={effective_contrib_u_y:.4f}, 比值={effective_contrib_u_y/effective_contrib_s_y:.4f}")

        # 3.9 严谨的方差分解（线性解码器下）
        print(f"\n  [3.9] 输出空间方差分解 (线性解码器，验证正交性到输出空间):")

        # 计算 x_s = W_s @ s 和 x_u = W_u @ u
        # shared_x_np: (N, shared_dim), W_s_x.T: (shared_dim, dim_x)
        x_s = shared_x_np @ W_s_x.T  # (N, dim_x)
        x_u = unique_x_np @ W_u_x.T  # (N, dim_x)
        y_s = shared_y_np @ W_s_y.T  # (N, dim_y)
        y_u = unique_y_np @ W_u_y.T  # (N, dim_y)

        # 方差（使用 ddof=1 无偏估计）
        var_x_s = np.var(x_s, ddof=1)
        var_x_u = np.var(x_u, ddof=1)
        var_y_s = np.var(y_s, ddof=1)
        var_y_u = np.var(y_u, ddof=1)

        # 协方差
        # Cov(x_s, x_u) = E[(x_s - mean)(x_u - mean)^T] 的迹 / dim
        x_s_centered = x_s - x_s.mean(axis=0)
        x_u_centered = x_u - x_u.mean(axis=0)
        y_s_centered = y_s - y_s.mean(axis=0)
        y_u_centered = y_u - y_u.mean(axis=0)

        # 逐元素协方差的平均
        cov_x_s_u = np.mean(x_s_centered * x_u_centered)  # 标量
        cov_y_s_u = np.mean(y_s_centered * y_u_centered)

        print(f"    [X 空间]:")
        print(f"      Var(x_s) = {var_x_s:.6f}")
        print(f"      Var(x_u) = {var_x_u:.6f}")
        print(f"      Cov(x_s, x_u) = {cov_x_s_u:.6f}")
        print(f"      理论: Var(x_s) + Var(x_u) + 2*Cov = {var_x_s + var_x_u + 2*cov_x_s_u:.6f}")
        print(f"      实际 Var(X_scaled) = {var_x:.6f}")
        print(f"      shared 贡献比例: {var_x_s / var_x * 100:.2f}%")
        print(f"      unique 贡献比例: {var_x_u / var_x * 100:.2f}%")
        print(f"      协方差贡献比例: {2*cov_x_s_u / var_x * 100:.2f}%")

        print(f"    [Y 空间]:")
        print(f"      Var(y_s) = {var_y_s:.6f}")
        print(f"      Var(y_u) = {var_y_u:.6f}")
        print(f"      Cov(y_s, y_u) = {cov_y_s_u:.6f}")
        print(f"      理论: Var(y_s) + Var(y_u) + 2*Cov = {var_y_s + var_y_u + 2*cov_y_s_u:.6f}")
        print(f"      实际 Var(Y_scaled) = {var_y:.6f}")
        print(f"      shared 贡献比例: {var_y_s / var_y * 100:.2f}%")
        print(f"      unique 贡献比例: {var_y_u / var_y * 100:.2f}%")
        print(f"      协方差贡献比例: {2*cov_y_s_u / var_y * 100:.2f}%")

        # 输出空间的 shared-unique 正交性（Frobenius 范数）
        cross_cov_output_x = x_s_centered.T @ x_u_centered / (n_samples - 1)
        cross_cov_output_y = y_s_centered.T @ y_u_centered / (n_samples - 1)
        ortho_output_x = np.linalg.norm(cross_cov_output_x, 'fro') ** 2
        ortho_output_y = np.linalg.norm(cross_cov_output_y, 'fro') ** 2

        print(f"    [输出空间正交性]:")
        print(f"      X: ||Cov(x_s, x_u)||_F² = {ortho_output_x:.6f}")
        print(f"      Y: ||Cov(y_s, y_u)||_F² = {ortho_output_y:.6f}")

    # 可视化
    if plot_path is not None:
        print(f"\n  [Step 4] 生成可视化图表...")

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        # 1. 训练损失曲线
        ax = axes[0, 0]
        ax.plot(history['total_loss'], label='Total Loss', linewidth=2)
        ax.plot(history['recon_loss'], label='Recon Loss', linewidth=2)
        ax.plot(history['shared_recon_loss'], label='Shared Recon Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. VICReg 相关损失
        ax = axes[0, 1]
        ax.plot(history['vicreg_loss'], label='VICReg Total', linewidth=2)
        ax.plot(history['vicreg_var'], label='Variance Loss', linewidth=2)
        ax.plot(history['ortho_loss'], label='Orthogonality Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('VICReg & Orthogonality Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Shared 空间分布
        ax = axes[0, 2]
        ax.scatter(shared_x_np[:, 0], shared_x_np[:, 1], alpha=0.5, label='VLM Shared', s=10)
        ax.scatter(shared_y_np[:, 0], shared_y_np[:, 1], alpha=0.5, label='ViT Shared', s=10)
        ax.set_xlabel('Shared Dim 1')
        ax.set_ylabel('Shared Dim 2')
        ax.set_title('Shared Space (First 2 Dims)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 方差解释比例（包括跨模态）
        ax = axes[1, 0]
        categories = ['X Full', 'X Shared\n(self)', 'X Shared\n(cross)', 'Y Full', 'Y Shared\n(self)', 'Y Shared\n(cross)']
        r2_values_all = [r2_full_x, r2_shared_x, r2_x_from_shared_y, r2_full_y, r2_shared_y, r2_y_from_shared_x]
        colors = ['steelblue', 'lightblue', 'skyblue', 'coral', 'lightsalmon', 'peachpuff']
        bars = ax.bar(categories, r2_values_all, color=colors)
        ax.set_ylabel('R² (Variance Explained)')
        ax.set_title('Reconstruction Quality (incl. Cross-Modal)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, r2_values_all):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # 5. Shared 各维方差
        ax = axes[1, 1]
        ax.hist(shared_x_std, bins=20, alpha=0.5, label='VLM Shared std')
        ax.hist(shared_y_std, bins=20, alpha=0.5, label='ViT Shared std')
        ax.axvline(x=1.0, color='r', linestyle='--', label='Target std=1')
        ax.set_xlabel('Std per Dimension')
        ax.set_ylabel('Count')
        ax.set_title('Shared Encoding Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. CKA 对比
        ax = axes[1, 2]
        cka_names = ['Original', 'Shared']
        cka_values = [cka_original, cka_shared]
        if cka_unique is not None:
            cka_names.append('Unique')
            cka_values.append(cka_unique)
        bars = ax.bar(cka_names, cka_values, color=['gray', 'green', 'orange'][:len(cka_names)])
        ax.set_ylabel('CKA Similarity')
        ax.set_title('CKA Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, cka_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom')

        # 7. 解码器权重诊断（如果是线性解码器）
        ax = axes[2, 0]
        if use_linear_decoder:
            weight_names = ['X W_s', 'X W_u', 'Y W_s', 'Y W_u']
            weight_norms = [norm_W_s_x, norm_W_u_x, norm_W_s_y, norm_W_u_y]
            colors_w = ['steelblue', 'lightblue', 'coral', 'lightsalmon']
            bars = ax.bar(weight_names, weight_norms, color=colors_w)
            ax.set_ylabel('Frobenius Norm')
            ax.set_title('Decoder Weight Norms')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, weight_norms):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.2f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'N/A\n(Non-linear decoder)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decoder Weight Norms')

        # 8. 输出空间方差分解（如果是线性解码器）
        ax = axes[2, 1]
        if use_linear_decoder:
            x_vals = [0, 1]
            width = 0.35
            ax.bar([x - width/2 for x in x_vals], [var_x_s/var_x*100, var_y_s/var_y*100], width, label='Shared', color='green', alpha=0.7)
            ax.bar([x + width/2 for x in x_vals], [var_x_u/var_x*100, var_y_u/var_y*100], width, label='Unique', color='orange', alpha=0.7)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(['X', 'Y'])
            ax.set_ylabel('Variance Contribution (%)')
            ax.set_title('Output Space Variance Decomposition')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'N/A\n(Non-linear decoder)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Output Space Variance Decomposition')

        # 9. 跨模态 vs 自身对比
        ax = axes[2, 2]
        comparison_names = ['X self', 'X cross', 'Y self', 'Y cross']
        comparison_values = [r2_shared_x, r2_x_from_shared_y, r2_shared_y, r2_y_from_shared_x]
        colors_comp = ['steelblue', 'lightsteelblue', 'coral', 'mistyrose']
        bars = ax.bar(comparison_names, comparison_values, color=colors_comp)
        ax.set_ylabel('R² (Shared Only)')
        ax.set_title('Self vs Cross-Modal Shared Reconstruction')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, comparison_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"    图表已保存到: {plot_path}")

    print(f"\n  {'='*70}")
    print(f"  [Summary] SAE 分析摘要")
    print(f"  {'='*70}")
    print(f"    架构:")
    print(f"      - 共享维度: {shared_dim}")
    print(f"      - X 特有维度: {unique_x_dim}")
    print(f"      - Y 特有维度: {unique_y_dim}")
    print(f"      - 解码器: {'线性可加' if use_linear_decoder else '非线性'}")
    print(f"    重构质量 (R²):")
    print(f"      - X 完整: {r2_full_x:.4f}, 仅shared(自身): {r2_shared_x:.4f}, 仅shared(跨模态): {r2_x_from_shared_y:.4f}")
    print(f"      - Y 完整: {r2_full_y:.4f}, 仅shared(自身): {r2_shared_y:.4f}, 仅shared(跨模态): {r2_y_from_shared_x:.4f}")
    print(f"    跨模态验证:")
    print(f"      - X: 自身shared R²={r2_shared_x:.4f} vs 跨模态shared R²={r2_x_from_shared_y:.4f} (差异={r2_shared_x - r2_x_from_shared_y:.4f})")
    print(f"      - Y: 自身shared R²={r2_shared_y:.4f} vs 跨模态shared R²={r2_y_from_shared_x:.4f} (差异={r2_shared_y - r2_y_from_shared_x:.4f})")
    print(f"    Shared 空间:")
    print(f"      - 对齐 MSE: {shared_align_mse:.6f}")
    print(f"      - 余弦相似度: {cosine_sim_mean:.6f}")
    print(f"      - CKA: {cka_shared:.6f}")
    print(f"    正交性 (latent 空间，越小越好):")
    print(f"      - X: {ortho_x:.6f}")
    print(f"      - Y: {ortho_y:.6f}")
    if use_linear_decoder:
        print(f"    方差贡献 (输出空间):")
        print(f"      - X: shared={var_x_s/var_x*100:.2f}%, unique={var_x_u/var_x*100:.2f}%, cov={2*cov_x_s_u/var_x*100:.2f}%")
        print(f"      - Y: shared={var_y_s/var_y*100:.2f}%, unique={var_y_u/var_y*100:.2f}%, cov={2*cov_y_s_u/var_y*100:.2f}%")

    # 返回关键指标（用于批量实验汇总）
    if return_metrics:
        metrics = {
            # 重构质量
            'r2_full_x': r2_full_x,
            'r2_full_y': r2_full_y,
            'r2_shared_x': r2_shared_x,
            'r2_shared_y': r2_shared_y,
            # 跨模态重构
            'r2_x_from_shared_y': r2_x_from_shared_y,
            'r2_y_from_shared_x': r2_y_from_shared_x,
            'r2_x_mixed': r2_x_mixed,
            'r2_y_mixed': r2_y_mixed,
            # 跨模态差异
            'cross_diff_x': r2_shared_x - r2_x_from_shared_y,
            'cross_diff_y': r2_shared_y - r2_y_from_shared_x,
            # Shared 空间对齐
            'shared_align_mse': shared_align_mse,
            'shared_cosine_sim': cosine_sim_mean,
            'cka_shared': cka_shared,
            'cka_original': cka_original,
            # 正交性
            'ortho_x': ortho_x,
            'ortho_y': ortho_y,
        }
        if use_linear_decoder:
            # 方差贡献比例
            var_x_cov_pct = 2 * cov_x_s_u / var_x * 100
            var_y_cov_pct = 2 * cov_y_s_u / var_y * 100
            # 残差比例 = 100 - shared - unique - cov（理论上应为0，实际可能有偏差）
            var_x_resid_pct = 100 - (var_x_s / var_x * 100) - (var_x_u / var_x * 100) - var_x_cov_pct
            var_y_resid_pct = 100 - (var_y_s / var_y * 100) - (var_y_u / var_y * 100) - var_y_cov_pct

            metrics.update({
                'var_x_shared_pct': var_x_s / var_x * 100,
                'var_x_unique_pct': var_x_u / var_x * 100,
                'var_x_cov_pct': var_x_cov_pct,
                'var_x_resid_pct': var_x_resid_pct,
                'var_y_shared_pct': var_y_s / var_y * 100,
                'var_y_unique_pct': var_y_u / var_y * 100,
                'var_y_cov_pct': var_y_cov_pct,
                'var_y_resid_pct': var_y_resid_pct,
            })
        if cka_unique is not None:
            metrics['cka_unique'] = cka_unique
        return metrics

    return None


def load_and_evaluate_model(
    model_path: str,
    X: np.ndarray,
    Y: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    加载已训练的模型并计算评估指标

    参数：
    - model_path: 模型文件路径
    - X, Y: 特征矩阵（原始空间）
    - device: 计算设备

    返回：
    - 评估指标字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    # The historical checkpoints contain fitted scikit-learn scalers.
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

    # 标准化数据
    X_scaled = model.scaler_x.transform(X).astype(np.float32)
    Y_scaled = model.scaler_y.transform(Y).astype(np.float32)
    n_samples = X.shape[0]

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        Y_tensor = torch.from_numpy(Y_scaled).to(device)

        # 编码
        shared_x = model.shared_encoder_x(X_tensor)
        shared_y = model.shared_encoder_y(Y_tensor)
        unique_x = model.unique_encoder_x(X_tensor)
        unique_y = model.unique_encoder_y(Y_tensor)

        # 完整重构
        x_recon = model.decode_x(shared_x, unique_x)
        y_recon = model.decode_y(shared_y, unique_y)

        # 仅 shared 重构
        x_shared_recon = model.decode_x_shared_only(shared_x)
        y_shared_recon = model.decode_y_shared_only(shared_y)

        # 跨模态重构
        x_from_shared_y_only = model.decode_x_shared_only(shared_y)
        y_from_shared_x_only = model.decode_y_shared_only(shared_x)
        x_from_shared_y_unique_x = model.decode_x(shared_y, unique_x)
        y_from_shared_x_unique_y = model.decode_y(shared_x, unique_y)

        # 转为 numpy
        shared_x_np = shared_x.cpu().numpy()
        shared_y_np = shared_y.cpu().numpy()
        unique_x_np = unique_x.cpu().numpy()
        unique_y_np = unique_y.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()
        y_recon_np = y_recon.cpu().numpy()
        x_shared_recon_np = x_shared_recon.cpu().numpy()
        y_shared_recon_np = y_shared_recon.cpu().numpy()
        x_from_shared_y_only_np = x_from_shared_y_only.cpu().numpy()
        y_from_shared_x_only_np = y_from_shared_x_only.cpu().numpy()
        x_from_shared_y_unique_x_np = x_from_shared_y_unique_x.cpu().numpy()
        y_from_shared_x_unique_y_np = y_from_shared_x_unique_y.cpu().numpy()

    # 计算指标（使用 ddof=1 无偏估计，与方差分解一致）
    var_x = np.var(X_scaled, ddof=1)
    var_y = np.var(Y_scaled, ddof=1)

    # 重构 MSE
    recon_mse_x = np.mean((X_scaled - x_recon_np) ** 2)
    recon_mse_y = np.mean((Y_scaled - y_recon_np) ** 2)
    shared_recon_mse_x = np.mean((X_scaled - x_shared_recon_np) ** 2)
    shared_recon_mse_y = np.mean((Y_scaled - y_shared_recon_np) ** 2)
    cross_mse_x_from_shared_y = np.mean((X_scaled - x_from_shared_y_only_np) ** 2)
    cross_mse_y_from_shared_x = np.mean((Y_scaled - y_from_shared_x_only_np) ** 2)
    cross_mse_x_mixed = np.mean((X_scaled - x_from_shared_y_unique_x_np) ** 2)
    cross_mse_y_mixed = np.mean((Y_scaled - y_from_shared_x_unique_y_np) ** 2)

    # R²
    r2_full_x = 1 - recon_mse_x / var_x
    r2_full_y = 1 - recon_mse_y / var_y
    r2_shared_x = 1 - shared_recon_mse_x / var_x
    r2_shared_y = 1 - shared_recon_mse_y / var_y
    r2_x_from_shared_y = 1 - cross_mse_x_from_shared_y / var_x
    r2_y_from_shared_x = 1 - cross_mse_y_from_shared_x / var_y
    r2_x_mixed = 1 - cross_mse_x_mixed / var_x
    r2_y_mixed = 1 - cross_mse_y_mixed / var_y

    # Shared 空间对齐
    shared_x_norm = (shared_x_np - shared_x_np.mean(axis=0)) / (shared_x_np.std(axis=0) + 1e-6)
    shared_y_norm = (shared_y_np - shared_y_np.mean(axis=0)) / (shared_y_np.std(axis=0) + 1e-6)
    shared_align_mse = np.mean((shared_x_norm - shared_y_norm) ** 2)

    # 余弦相似度
    cosine_sims = []
    for i in range(n_samples):
        norm_x = np.linalg.norm(shared_x_np[i])
        norm_y = np.linalg.norm(shared_y_np[i])
        if norm_x > 1e-8 and norm_y > 1e-8:
            cosine_sims.append(np.dot(shared_x_np[i], shared_y_np[i]) / (norm_x * norm_y))
    cosine_sim_mean = np.mean(cosine_sims) if cosine_sims else 0

    # CKA
    cka_shared = linear_cka(shared_x_np, shared_y_np)
    cka_original = linear_cka(X, Y)

    # 正交性
    S_x = shared_x_np - shared_x_np.mean(axis=0)
    U_x = unique_x_np - unique_x_np.mean(axis=0)
    cross_cov_x = S_x.T @ U_x / (n_samples - 1)
    ortho_x = np.linalg.norm(cross_cov_x, 'fro') ** 2

    S_y = shared_y_np - shared_y_np.mean(axis=0)
    U_y = unique_y_np - unique_y_np.mean(axis=0)
    cross_cov_y = S_y.T @ U_y / (n_samples - 1)
    ortho_y = np.linalg.norm(cross_cov_y, 'fro') ** 2

    # 构建指标字典
    metrics = {
        'r2_full_x': r2_full_x,
        'r2_full_y': r2_full_y,
        'r2_shared_x': r2_shared_x,
        'r2_shared_y': r2_shared_y,
        'r2_x_from_shared_y': r2_x_from_shared_y,
        'r2_y_from_shared_x': r2_y_from_shared_x,
        'r2_x_mixed': r2_x_mixed,
        'r2_y_mixed': r2_y_mixed,
        'cross_diff_x': r2_shared_x - r2_x_from_shared_y,
        'cross_diff_y': r2_shared_y - r2_y_from_shared_x,
        'shared_align_mse': shared_align_mse,
        'shared_cosine_sim': cosine_sim_mean,
        'cka_shared': cka_shared,
        'cka_original': cka_original,
        'ortho_x': ortho_x,
        'ortho_y': ortho_y,
    }

    # 线性解码器的方差分解
    if config['use_linear_decoder']:
        W_s_x = model.decoder_x_shared.weight.detach().cpu().numpy()
        W_u_x = model.decoder_x_unique.weight.detach().cpu().numpy()
        W_s_y = model.decoder_y_shared.weight.detach().cpu().numpy()
        W_u_y = model.decoder_y_unique.weight.detach().cpu().numpy()

        x_s = shared_x_np @ W_s_x.T
        x_u = unique_x_np @ W_u_x.T
        y_s = shared_y_np @ W_s_y.T
        y_u = unique_y_np @ W_u_y.T

        var_x_s = np.var(x_s, ddof=1)
        var_x_u = np.var(x_u, ddof=1)
        var_y_s = np.var(y_s, ddof=1)
        var_y_u = np.var(y_u, ddof=1)

        x_s_centered = x_s - x_s.mean(axis=0)
        x_u_centered = x_u - x_u.mean(axis=0)
        y_s_centered = y_s - y_s.mean(axis=0)
        y_u_centered = y_u - y_u.mean(axis=0)

        cov_x_s_u = np.mean(x_s_centered * x_u_centered)
        cov_y_s_u = np.mean(y_s_centered * y_u_centered)

        # 方差贡献比例
        var_x_shared_pct = var_x_s / var_x * 100
        var_x_unique_pct = var_x_u / var_x * 100
        var_x_cov_pct = 2 * cov_x_s_u / var_x * 100
        var_x_resid_pct = 100 - var_x_shared_pct - var_x_unique_pct - var_x_cov_pct

        var_y_shared_pct = var_y_s / var_y * 100
        var_y_unique_pct = var_y_u / var_y * 100
        var_y_cov_pct = 2 * cov_y_s_u / var_y * 100
        var_y_resid_pct = 100 - var_y_shared_pct - var_y_unique_pct - var_y_cov_pct

        metrics.update({
            'var_x_shared_pct': var_x_shared_pct,
            'var_x_unique_pct': var_x_unique_pct,
            'var_x_cov_pct': var_x_cov_pct,
            'var_x_resid_pct': var_x_resid_pct,
            'var_y_shared_pct': var_y_shared_pct,
            'var_y_unique_pct': var_y_unique_pct,
            'var_y_cov_pct': var_y_cov_pct,
            'var_y_resid_pct': var_y_resid_pct,
        })

    # Unique CKA
    if config['unique_x_dim'] == config['unique_y_dim']:
        metrics['cka_unique'] = linear_cka(unique_x_np, unique_y_np)

    return metrics


def run_eval_from_saved_models(
    feature_configs: List[Tuple[str, np.ndarray, np.ndarray]],
    model_dir: str,
    output_csv: str,
    device: Optional[torch.device] = None,
    is_random: bool = False,
) -> pd.DataFrame:
    """Recompute metrics for the saved experiment grid."""
    raw_mse_options = [False, True]
    cross_weight_options = [0.0, 0.1, 0.2, 0.5, 1.0]

    all_results = []
    total_experiments = len(raw_mse_options) * len(cross_weight_options) * len(feature_configs)
    exp_idx = 0

    print(f"\n{'#'*80}")
    print(f"# 从已保存模型重新计算指标")
    print(f"# 模型目录: {model_dir}")
    print(f"# 总实验数: {total_experiments}")
    print(f"{'#'*80}")

    for feature_name, X, Y in feature_configs:
        for use_raw_mse in raw_mse_options:
            for cross_weight in cross_weight_options:
                exp_idx += 1

                # 生成文件名（与训练时一致）
                exp_suffix = f"_{feature_name}" + ("_random" if is_random else "")
                if use_raw_mse:
                    exp_suffix += "_rawMSE"
                if cross_weight > 0:
                    exp_suffix += f"_crossRecon{cross_weight}"
                else:
                    exp_suffix += "_noCross"

                model_path = os.path.join(model_dir, f"sae_model{exp_suffix}.pt")

                print(f"\n[{exp_idx}/{total_experiments}] 加载: {model_path}")

                if not os.path.exists(model_path):
                    print(f"  警告: 模型文件不存在，跳过")
                    continue

                try:
                    metrics = load_and_evaluate_model(model_path, X, Y, device)

                    result = {
                        'feature': feature_name,
                        'is_random': is_random,
                        'use_raw_mse': use_raw_mse,
                        'cross_weight': cross_weight,
                        **metrics
                    }
                    all_results.append(result)

                    print(f"  R² full: X={metrics['r2_full_x']:.4f}, Y={metrics['r2_full_y']:.4f}")
                    print(f"  Cross diff: X={metrics['cross_diff_x']:.4f}, Y={metrics['cross_diff_y']:.4f}")
                    if 'var_x_cov_pct' in metrics:
                        print(f"  Var cov%: X={metrics['var_x_cov_pct']:.2f}%, Y={metrics['var_y_cov_pct']:.2f}%")
                        print(f"  Var resid%: X={metrics['var_x_resid_pct']:.2f}%, Y={metrics['var_y_resid_pct']:.2f}%")

                except Exception as e:
                    print(f"  错误: {e}")
                    continue

    # 创建 DataFrame 并保存
    if not all_results:
        raise ValueError(f"No SAE models could be evaluated from {model_dir}.")
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")

    # 打印汇总
    print_summary_table(all_results)

    return df


def run_batch_experiments(
    X: np.ndarray,
    Y: np.ndarray,
    feature_name: str,
    output_dir: str,
    shared_dim: int = 64,
    unique_x_dim: int = 16,
    unique_y_dim: int = 16,
    hidden_dim: int = 256,
    use_linear_decoder: bool = True,
    batch_size: int = 128,
    num_epochs: int = 150,
    learning_rate: float = 1e-3,
    recon_weight: float = 1.0,
    shared_recon_weight: float = 1.0,
    vicreg_weight: float = 1.0,
    ortho_weight: float = 0.1,
    unique_sparse_weight: float = 0.1,
    device: Optional[torch.device] = None,
    is_random: bool = False,
) -> List[Dict]:
    """
    批量运行多组实验

    参数：
    - X, Y: 特征矩阵
    - feature_name: 特征名称（用于区分 backbone/dit）
    - output_dir: 输出目录
    - is_random: 是否为打乱配对的基线实验
    - 其他参数: SAE 配置

    返回：
    - 所有实验结果的列表
    """
    # 实验参数组合
    raw_mse_options = [False, True]
    cross_weight_options = [0.0, 0.1, 0.2, 0.5, 1.0]

    results = []
    total_experiments = len(raw_mse_options) * len(cross_weight_options)
    exp_idx = 0

    random_tag = " [RANDOM BASELINE]" if is_random else ""

    print(f"\n{'#'*80}")
    print(f"# 批量实验: {feature_name}{random_tag}")
    print(f"# 总实验数: {total_experiments}")
    print(f"{'#'*80}")

    for use_raw_mse in raw_mse_options:
        for cross_weight in cross_weight_options:
            exp_idx += 1

            print(f"\n{'='*80}")
            print(f"实验 {exp_idx}/{total_experiments}: "
                  f"feature={feature_name}, raw_mse={use_raw_mse}, cross_weight={cross_weight}{random_tag}")
            print(f"{'='*80}")

            # 生成文件名后缀
            exp_suffix = f"_{feature_name}"
            if is_random:
                exp_suffix += "_random"
            if use_raw_mse:
                exp_suffix += "_rawMSE"
            if cross_weight > 0:
                exp_suffix += f"_crossRecon{cross_weight}"
            else:
                exp_suffix += "_noCross"

            model_save_path = os.path.join(output_dir, f"sae_model{exp_suffix}.pt")
            plot_path = os.path.join(output_dir, f"sae_results{exp_suffix}.png")

            # 运行实验
            metrics = sae_analysis(
                X, Y,
                shared_dim=shared_dim,
                unique_x_dim=unique_x_dim,
                unique_y_dim=unique_y_dim,
                hidden_dim=hidden_dim,
                use_linear_decoder=use_linear_decoder,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                recon_weight=recon_weight,
                shared_recon_weight=shared_recon_weight,
                vicreg_weight=vicreg_weight,
                ortho_weight=ortho_weight,
                unique_sparse_weight=unique_sparse_weight,
                use_raw_mse_invariance=use_raw_mse,
                cross_shared_recon_weight=cross_weight,
                device=device,
                plot_path=plot_path,
                model_save_path=model_save_path,
                return_metrics=True,
            )

            # 记录实验配置和结果
            result = {
                'feature': feature_name,
                'is_random': is_random,
                'use_raw_mse': use_raw_mse,
                'cross_weight': cross_weight,
                **metrics
            }
            results.append(result)

    return results


def print_summary_table(all_results: List[Dict], output_path: Optional[str] = None):
    """
    打印汇总表格
    """
    # 创建 DataFrame
    df = pd.DataFrame(all_results)

    # 选择关键列（包含新增的 cov 和 resid 列）
    key_columns = [
        'feature', 'is_random', 'use_raw_mse', 'cross_weight',
        'r2_full_x', 'r2_full_y',
        'r2_shared_x', 'r2_shared_y',
        'r2_x_from_shared_y', 'r2_y_from_shared_x',
        'cross_diff_x', 'cross_diff_y',
        'shared_cosine_sim', 'cka_shared',
        'var_x_cov_pct', 'var_y_cov_pct',
        'var_x_resid_pct', 'var_y_resid_pct',
    ]

    # 只保留存在的列
    available_columns = [c for c in key_columns if c in df.columns]
    df_summary = df[available_columns].copy()

    # 格式化数值
    for col in df_summary.columns:
        if df_summary[col].dtype in [float, np.float64, np.float32]:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print("\n" + "="*120)
    print("批量实验汇总表")
    print("="*120)
    print(df_summary.to_string(index=False))
    print("="*120)

    # 保存完整结果到 CSV
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"\n完整结果已保存到: {output_path}")

    # 打印关键发现
    print("\n" + "-"*80)
    print("关键指标分析:")
    print("-"*80)

    # 按 feature 分组分析
    for feature in df['feature'].unique():
        df_feat = df[df['feature'] == feature]
        print(f"\n[{feature}]:")

        # 找到跨模态差异最小的配置
        best_cross_x = df_feat.loc[df_feat['cross_diff_x'].abs().idxmin()]
        best_cross_y = df_feat.loc[df_feat['cross_diff_y'].abs().idxmin()]

        print(f"  X 跨模态差异最小: raw_mse={best_cross_x['use_raw_mse']}, "
              f"cross_weight={best_cross_x['cross_weight']}, "
              f"diff={best_cross_x['cross_diff_x']:.4f}")
        print(f"  Y 跨模态差异最小: raw_mse={best_cross_y['use_raw_mse']}, "
              f"cross_weight={best_cross_y['cross_weight']}, "
              f"diff={best_cross_y['cross_diff_y']:.4f}")

        # Shared CKA 最高的配置
        best_cka = df_feat.loc[df_feat['cka_shared'].idxmax()]
        print(f"  Shared CKA 最高: raw_mse={best_cka['use_raw_mse']}, "
              f"cross_weight={best_cka['cross_weight']}, "
              f"cka={best_cka['cka_shared']:.4f}")

    return df


def main():
    import argparse
    from pathlib import Path

    from pair_inputs import add_pair_arguments, load_pair, parse_pair_arguments

    parser = argparse.ArgumentParser(description="Shared/unique SAE analysis for paired scene features.")
    add_pair_arguments(parser)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--single", action="store_true", help="Train one model (the default).")
    modes.add_argument("--batch", action="store_true", help="Run the full loss-weight sweep.")
    modes.add_argument("--eval", action="store_true", help="Recompute metrics using saved models.")
    parser.add_argument("--model-dir", type=Path, help="Trusted archive of trained sweep models.")
    parser.add_argument("--model", type=Path, help="Evaluate one trusted SAE checkpoint.")
    parser.add_argument("--feature-type", choices=("backbone", "dit"), default="backbone")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cross-weight", type=float, default=0.5)
    parser.add_argument("--raw-mse", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random", action="store_true", help="Shuffle Y to run the unpaired control.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--random-prefix-size", type=int, default=0,
        help="Consume a permutation of this size first, as in the historical DiT control.",
    )
    args = parse_pair_arguments(parser)
    if args.eval and bool(args.model) == bool(args.model_dir):
        parser.error("--eval requires exactly one of --model or --model-dir.")
    if (args.model or args.model_dir) and not args.eval:
        parser.error("--model and --model-dir require --eval.")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size must be at least 2.")

    if args.random_prefix_size < 0 or (args.random_prefix_size and not args.random):
        parser.error("--random-prefix-size must be nonnegative and requires --random.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X, Y, _ = load_pair(args.cache, args.x_dir, args.y_dir, args.max_scenes)
    print(f"Aligned samples: {len(X)}, X dims: {X.shape[1]}, Y dims: {Y.shape[1]}")
    print(f"Paired CKA: {linear_cka(X, Y):.6f}")
    if args.random:
        rng = np.random.RandomState(args.random_seed)
        if args.random_prefix_size:
            rng.permutation(args.random_prefix_size)
        permutation = rng.permutation(len(Y))
        Y = Y[permutation]
        print(f"Shuffled CKA: {linear_cka(X, Y):.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.feature_type}" + ("_random" if args.random else "")
    device = torch.device(args.device)
    if args.eval:
        if args.model:
            metrics = load_and_evaluate_model(str(args.model), X, Y, device)
            result = {"feature": args.feature_type, "is_random": args.random, **metrics}
            pd.DataFrame([result]).to_csv(args.output / f"sae_metrics{suffix}.csv", index=False)
        else:
            run_eval_from_saved_models(
                [(args.feature_type, X, Y)], str(args.model_dir),
                str(args.output / f"batch_experiment_results_updated{suffix}.csv"),
                device=device, is_random=args.random,
            )
        return

    training = {
        "shared_dim": 64, "unique_x_dim": 16, "unique_y_dim": 16, "hidden_dim": 256,
        "use_linear_decoder": True, "learning_rate": 1e-3, "recon_weight": 1.0,
        "shared_recon_weight": 1.0, "vicreg_weight": 1.0, "ortho_weight": 0.1,
        "unique_sparse_weight": 0.1, "batch_size": args.batch_size,
        "num_epochs": args.epochs, "device": device,
    }
    if args.batch:
        rows = run_batch_experiments(
            X, Y, args.feature_type, str(args.output), is_random=args.random, **training,
        )
        print_summary_table(rows, str(args.output / f"batch_experiment_results{suffix}.csv"))
        return

    if args.raw_mse:
        suffix += "_rawMSE"
    suffix += f"_crossRecon{args.cross_weight}" if args.cross_weight > 0 else "_noCross"
    sae_analysis(
        X, Y, use_raw_mse_invariance=args.raw_mse, cross_shared_recon_weight=args.cross_weight,
        plot_path=str(args.output / f"sae_results{suffix}.png"),
        model_save_path=str(args.output / f"sae_model{suffix}.pt"), **training,
    )


if __name__ == "__main__":
    main()
