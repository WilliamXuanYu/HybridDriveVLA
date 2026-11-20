# Recovered analysis code adapted for the ReCogDrive visual backbone ablation release.
"""多模型表征空间高级可视化 v2 - 优化版本"""
import os, numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import MinCovDet
from scipy import stats
from scipy.stats import chi2
from analyze_pca_whitening_cca import linear_cka
import warnings; warnings.filterwarnings('ignore')

try:
    import torch
    from analyze_sae_shared_unique import SharedUniqueSAE
    HAS_TORCH = True
except: HAS_TORCH = False

COLORS = {"VLM": "#E74C3C", "ViT-Large": "#3498DB", "ResNet101": "#2ECC71", "EVA-CLIP": "#9B59B6"}
NAMES = {"VLM": "VLM", "ViT-Large": "ViT-Large", "ResNet101": "ResNet-101", "EVA-CLIP": "EVA-CLIP"}

def fit_procrustes(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return U @ Vt

def align_all(feats, models, ref="ResNet101"):
    print(f"\n[Procrustes] ref={ref}")
    aligned, resid = {}, {}
    s = StandardScaler()
    ref_s = s.fit_transform(feats[ref])
    aligned[ref], resid[ref] = ref_s, np.zeros_like(ref_s)
    for m in models:
        if m == ref: continue
        src = StandardScaler().fit_transform(feats[m])
        W = fit_procrustes(src, ref_s)
        aligned[m] = src @ W
        resid[m] = aligned[m] - ref_s
        print(f"  {m}: MSE={np.mean((aligned[m]-ref_s)**2):.4f}")
    return aligned, resid

def cosine(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return np.sum(Xn * Yn, axis=1)

def auroc(m, r):
    return roc_auc_score(np.concatenate([np.ones(len(m)), np.zeros(len(r))]), np.concatenate([m, r]))

def cohens_d(x, y):
    """计算 Cohen's d 效应大小"""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, stat_func=np.median):
    """Bootstrap 置信区间"""
    stats_list = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats_list.append(stat_func(sample))
    lower = np.percentile(stats_list, (1-ci)/2*100)
    upper = np.percentile(stats_list, (1+ci)/2*100)
    return lower, upper

def recall_k(X, Y, ks=[1,10,50]):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Yn.T
    return {k: sum(i in np.argsort(-sim[i])[:k] for i in range(len(X)))/len(X) for k in ks}

def recall_k_curve(X, Y, max_k=200):
    """计算 K 从 1 到 max_k 的 Recall 曲线"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Yn.T
    n = len(X)
    recalls = []
    for k in range(1, max_k+1):
        r = sum(i in np.argsort(-sim[i])[:k] for i in range(n)) / n
        recalls.append(r)
    return recalls

def draw_ellipse95(ax, points, color, label, alpha=0.18):
    """基于鲁棒协方差 (MinCovDet) 的二维 95% 置信椭圆"""
    mcd = MinCovDet(random_state=0).fit(points)
    mean = mcd.location_
    cov = mcd.covariance_

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    scale = np.sqrt(chi2.ppf(0.95, df=2))  # 2D 95% 置信区域
    width, height = 2 * scale * np.sqrt(eigvals)

    ellipse = Ellipse(mean, width, height, angle=angle,
                      facecolor=color, edgecolor=color, lw=2, alpha=alpha)
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], c=color, s=120, marker='*',
               edgecolors='white', lw=1.5, zorder=10, label=label)


def kde_contours(ax, pts, color, gridsize=120, mass_levels=(0.50, 0.80, 0.95)):
    """按概率质量分位数绘制 KDE 等高线，避免尾部拉长影响"""
    kde = stats.gaussian_kde(pts.T, bw_method='scott')
    xmin, xmax = np.percentile(pts[:, 0], [0.5, 99.5])
    ymin, ymax = np.percentile(pts[:, 1], [0.5, 99.5])
    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    z = zz.ravel()
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted); cdf /= cdf[-1]
    thresh = [z_sorted[np.searchsorted(cdf, m)] for m in mass_levels]

    ax.contour(xx, yy, zz, levels=sorted(thresh), colors=[color],
               linewidths=1.4, alpha=0.75)

def plot_procrustes_residual(feats, models, path, ref="ResNet101", n=3000):
    """方案A: 左图 KDE 分位数等高线 + 鲁棒 95% 椭圆；右图 median+IQR+95%CI，限幅标注尾部"""
    print("\n[方案A] Procrustes+Residual (视觉优化版)")
    np.random.seed(42)
    aligned, resid = align_all(feats, models, ref)
    n = min(n, feats[models[0]].shape[0])
    idx = np.random.choice(feats[models[0]].shape[0], n, replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # === 左图: KDE 分位数等高线 + 鲁棒 95% 椭圆（无杂点） ===
    all_a = np.vstack([aligned[m][idx] for m in models])
    pca = PCA(2); coords = pca.fit_transform(all_a)
    mc = {m: coords[i*n:(i+1)*n] for i, m in enumerate(models)}

    draw_order = [m for m in models if m != "VLM"] + ["VLM"]  # VLM 最上层
    for m in draw_order:
        pts = mc[m]
        try:
            kde_contours(ax1, pts, COLORS[m], gridsize=140, mass_levels=(0.5, 0.6, 0.7))
        except Exception as e:
            print(f"  KDE 警告 {m}: {e}")
        draw_ellipse95(ax1, pts, COLORS[m], f'{NAMES[m]}', alpha=0.18)
        # 可选极少量散点，避免噪点，这里默认不画；如需提示可开启：
        scatter_idx = np.random.choice(len(pts), min(10, len(pts)), replace=False)
        ax1.scatter(pts[scatter_idx, 0], pts[scatter_idx, 1], c=COLORS[m], s=10, alpha=0.3, zorder=3)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax1.set_title(f'(a) Aligned Features (Ref: {ref})\nKDE mass contours (50/60/70%) + robust 95% ellipse', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # === 右图: 精简版 median + IQR + bootstrap 95% CI，限幅并标注尾部 ===
    data = [np.linalg.norm(resid[m][idx], axis=1) for m in models if m != ref]
    labels = [NAMES[m] for m in models if m != ref]
    colors = [COLORS[m] for m in models if m != ref]
    positions = range(1, len(labels)+1)

    # 精简，不画 violin，只画 IQR + CI
    # for i, d in enumerate(data):
    #     pos = i + 1
    #     med = np.median(d)
    #     q1, q3 = np.percentile(d, [25, 75])
    #     ci_low, ci_high = bootstrap_ci(d, n_bootstrap=1000, ci=0.95, stat_func=np.median)

    #     ax2.vlines(pos, q1, q3, color=colors[i], lw=8, alpha=0.7)  # IQR
    #     ax2.errorbar(pos, med, yerr=[[med-ci_low], [ci_high-med]], fmt='o',
    #                  color='black', capsize=6, markersize=7, zorder=5)

    # 右图：更直观的 quantile-interval plot
    for i, d in enumerate(data):
        pos = i + 1
        med = np.median(d)
        q10, q25, q75, q90 = np.percentile(d, [10, 25, 75, 90])

        ax2.vlines(pos, q25, q75, color=colors[i], lw=10, alpha=0.8, zorder=3)   # IQR
        ax2.vlines(pos, q10, q90, color=colors[i], lw=2.5, alpha=0.9, zorder=2)  # 10-90
        ax2.scatter(pos, med, s=70, color='white', edgecolors='black', lw=1.8, zorder=4)

    # 限幅 + 标注尾部 outliers
    all_d = np.concatenate(data)
    ymax = np.percentile(all_d, 99)  # 主体显示到 99 分位
    ax2.set_ylim(0, ymax)
    for i, d in enumerate(data):
        n_tail = np.sum(d > ymax)
        if n_tail > 0:
            ax2.text(i+1, ymax*0.98, f'+{n_tail} outliers', ha='center', va='top', fontsize=8)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Residual L2 Norm', fontsize=11)
    ax2.set_title('(b) Residual: Median ●, IQR | |, 95% CI (errorbar)\nY-lim clipped at 99%ile; tails annotated', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

def plot_cka_network(cka, models, path):
    print("\n[方案C] CKA网络图")
    fig, ax = plt.subplots(figsize=(10, 8))
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False) - np.pi/2
    pos = {models[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(len(models))}

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            c = cka[i,j]
            x = [pos[models[i]][0], pos[models[j]][0]]
            y = [pos[models[i]][1], pos[models[j]][1]]
            ax.plot(x, y, color=plt.cm.RdYlBu_r(c), lw=1+c*8, alpha=0.3+c*0.6)
            ax.text((x[0]+x[1])/2+0.08, (y[0]+y[1])/2+0.08, f'{c:.2f}', fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    for m in models:
        ax.scatter(*pos[m], c=COLORS[m], s=800, edgecolors='white', lw=3, zorder=3)
        ax.text(pos[m][0]*1.25, pos[m][1]*1.25, NAMES[m], fontsize=12, fontweight='bold', ha='center', color=COLORS[m])

    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('CKA Similarity Network\n(Edge Thickness ∝ CKA)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  保存: {path}")

def plot_auroc(feats, models, path, ref="ResNet101", n=5000):
    """方案D1: AUROC + Cohen's d + inset CDF"""
    print("\n[方案D1] 对齐后AUROC (优化版: +Cohen's d, +CDF inset)")
    np.random.seed(42)
    aligned, _ = align_all(feats, models, ref)
    n = min(n, feats[models[0]].shape[0])
    idx = np.random.choice(feats[models[0]].shape[0], n, replace=False)

    visual = [m for m in models if m != "VLM"]
    results = []  # (label, auroc, cohen_d, matched, random, color)

    for m in visual:
        matched = cosine(aligned["VLM"][idx], aligned[m][idx])
        rand = cosine(aligned["VLM"][idx], aligned[m][idx][np.random.permutation(n)])
        au = auroc(matched, rand)
        cd = cohens_d(matched, rand)
        results.append((f"VLM↔{m[:3]}", au, cd, matched, rand, '#E74C3C'))

    for i, a in enumerate(visual):
        for b in visual[i+1:]:
            matched = cosine(aligned[a][idx], aligned[b][idx])
            rand = cosine(aligned[a][idx], aligned[b][idx][np.random.permutation(n)])
            au = auroc(matched, rand)
            cd = cohens_d(matched, rand)
            results.append((f"{a[:3]}↔{b[:3]}", au, cd, matched, rand, '#3498DB'))

    fig, ax = plt.subplots(figsize=(14, 7))

    # 主图: AUROC 条形图
    labels = [r[0] for r in results]
    aurocs = [r[1] for r in results]
    cohens = [r[2] for r in results]
    colors = [r[5] for r in results]

    bars = ax.bar(range(len(aurocs)), aurocs, color=colors, alpha=0.8, edgecolor='white', lw=1)

    for i, (bar, au, cd) in enumerate(zip(bars, aurocs, cohens)):
        # AUROC 值
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{au:.3f}', ha='center', fontsize=10, fontweight='bold')
        # Cohen's d
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'd={cd:.2f}', ha='center', fontsize=8, color='white', fontweight='bold')

    ax.axhline(0.5, color='gray', ls='--', lw=1.5, label='Chance (0.5)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_ylim(0.4, 1.08)
    ax.set_title('Matched vs Random AUROC (Aligned Space)\nBar labels: AUROC, White text: Cohen\'s d effect size',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(facecolor='#E74C3C', label='VLM vs Visual'),
                       mpatches.Patch(facecolor='#3498DB', label='Visual vs Visual'),
                       Line2D([0], [0], color='gray', ls='--', label='Chance')],
              loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Inset: CDF 对比 (选择第一个 VLM pair)
    axins = inset_axes(ax, width="35%", height="40%", loc='upper left', borderpad=2)
    matched_ex, rand_ex = results[0][3], results[0][4]
    axins.plot(np.sort(matched_ex), np.linspace(0, 1, len(matched_ex)),
               color='#E74C3C', lw=2, label='Matched')
    axins.plot(np.sort(rand_ex), np.linspace(0, 1, len(rand_ex)),
               color='gray', lw=2, label='Random')
    axins.axvline(np.mean(matched_ex), color='#E74C3C', ls=':', alpha=0.7)
    axins.axvline(np.mean(rand_ex), color='gray', ls=':', alpha=0.7)
    delta_mean = np.mean(matched_ex) - np.mean(rand_ex)
    axins.text(0.5, 0.15, f'Δμ={delta_mean:.3f}', transform=axins.transAxes, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8))
    axins.set_xlabel('Cosine Sim', fontsize=8)
    axins.set_ylabel('CDF', fontsize=8)
    axins.set_title(f'CDF: {results[0][0]}', fontsize=9)
    axins.legend(fontsize=7, loc='lower right')
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

def plot_recall(feats, models, path, ref="ResNet101", n=5000, max_k=200):
    """方案D2: Recall@K 曲线 + 每对详细表格"""
    print("\n[方案D2] 对齐后Recall@K (优化版: +曲线, +详细pair)")
    np.random.seed(42)
    aligned, _ = align_all(feats, models, ref)
    n = min(n, feats[models[0]].shape[0])
    idx = np.random.choice(feats[models[0]].shape[0], n, replace=False)

    visual = [m for m in models if m != "VLM"]
    ks = [1, 5, 10, 20, 50, 100]

    # 收集所有 pair 的数据
    pair_data = {}

    # VLM vs Visual (双向)
    for m in visual:
        # VLM -> Visual
        name1 = f"VLM→{NAMES[m][:3]}"
        pair_data[name1] = {'curve': recall_k_curve(aligned["VLM"][idx], aligned[m][idx], max_k),
                           'color': '#E74C3C', 'ls': '-'}
        # Visual -> VLM
        name2 = f"{NAMES[m][:3]}→VLM"
        pair_data[name2] = {'curve': recall_k_curve(aligned[m][idx], aligned["VLM"][idx], max_k),
                           'color': '#E74C3C', 'ls': '--'}

    # Visual vs Visual
    for i, a in enumerate(visual):
        for b in visual[i+1:]:
            name = f"{NAMES[a][:3]}↔{NAMES[b][:3]}"
            pair_data[name] = {'curve': recall_k_curve(aligned[a][idx], aligned[b][idx], max_k),
                              'color': '#3498DB', 'ls': '-'}

    fig = plt.figure(figsize=(16, 10))

    # === 左上: Recall-K 曲线 (全范围) ===
    ax1 = fig.add_subplot(2, 2, 1)
    ks_range = np.arange(1, max_k+1)

    for name, data in pair_data.items():
        ax1.plot(ks_range, np.array(data['curve'])*100,
                color=data['color'], ls=data['ls'], lw=2, alpha=0.8, label=name)

    ax1.set_xlabel('K', fontsize=11)
    ax1.set_ylabel('Recall@K (%)', fontsize=11)
    ax1.set_title(f'(a) Recall-K Curves (K=1 to {max_k})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max_k)
    ax1.set_ylim(0, 105)

    # === 右上: 小 K 区间放大 ===
    ax2 = fig.add_subplot(2, 2, 2)
    for name, data in pair_data.items():
        ax2.plot(ks_range[:50], np.array(data['curve'][:50])*100,
                color=data['color'], ls=data['ls'], lw=2, alpha=0.8, label=name)
    ax2.set_xlabel('K', fontsize=11)
    ax2.set_ylabel('Recall@K (%)', fontsize=11)
    ax2.set_title('(b) Zoom: K=1 to 50', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 50)

    # === 左下: VLM 方向性对比 ===
    ax3 = fig.add_subplot(2, 2, 3)
    bar_labels = []
    bar_vals_fwd = []  # VLM → Visual
    bar_vals_bwd = []  # Visual → VLM

    for m in visual:
        bar_labels.append(NAMES[m][:3])
        fwd_name = f"VLM→{NAMES[m][:3]}"
        bwd_name = f"{NAMES[m][:3]}→VLM"
        # R@10 作为代表
        bar_vals_fwd.append(pair_data[fwd_name]['curve'][9] * 100)
        bar_vals_bwd.append(pair_data[bwd_name]['curve'][9] * 100)

    x = np.arange(len(bar_labels))
    w = 0.35
    ax3.bar(x - w/2, bar_vals_fwd, w, label='VLM → Visual', color='#E74C3C', alpha=0.8)
    ax3.bar(x + w/2, bar_vals_bwd, w, label='Visual → VLM', color='#E74C3C', alpha=0.4, hatch='//')

    for i, (v1, v2) in enumerate(zip(bar_vals_fwd, bar_vals_bwd)):
        ax3.text(i - w/2, v1 + 1, f'{v1:.1f}%', ha='center', fontsize=9)
        ax3.text(i + w/2, v2 + 1, f'{v2:.1f}%', ha='center', fontsize=9)
        # 标注差异
        diff = v1 - v2
        ax3.annotate('', xy=(i + w/2, max(v1, v2) + 5), xytext=(i - w/2, max(v1, v2) + 5),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1))
        ax3.text(i, max(v1, v2) + 7, f'Δ={diff:+.1f}', ha='center', fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(bar_labels)
    ax3.set_ylabel('R@10 (%)', fontsize=11)
    ax3.set_title('(c) Directionality: VLM ↔ Visual at R@10', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # === 右下: 详细表格 ===
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # 构建表格数据
    table_cols = ['Pair'] + [f'R@{k}' for k in ks]
    table_data = []
    for name, data in pair_data.items():
        row = [name] + [f"{data['curve'][k-1]*100:.1f}%" for k in ks]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=table_cols, loc='center',
                      cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 设置表头颜色
    for j in range(len(table_cols)):
        table[(0, j)].set_facecolor('#4A90D9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 设置行颜色
    for i, name in enumerate(pair_data.keys()):
        color = '#FFE4E1' if 'VLM' in name else '#E6F3FF'
        for j in range(len(table_cols)):
            table[(i+1, j)].set_facecolor(color)

    ax4.set_title('(d) Detailed Recall@K for All Pairs', fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

def plot_sae(feats, sae_dir, path, pair=("VLM", "ViT-Large"), n=3000, device=None):
    """方案B: SAE 可视化优化版"""
    if not HAS_TORCH: print("[方案B] 跳过: PyTorch未安装"); return
    print("\n[方案B] SAE机制化 (优化版)")
    np.random.seed(42)
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    short = {"VLM": "VLM", "ViT-Large": "ViT", "ResNet101": "ResNet", "EVA-CLIP": "EVA"}
    p = os.path.join(sae_dir, f"sae_model_{short[pair[0]]}_{short[pair[1]]}.pt")
    if not os.path.exists(p): print(f"  模型不存在: {p}"); return

    ckpt = torch.load(p, map_location=device, weights_only=False); cfg = ckpt['config']
    model = SharedUniqueSAE(cfg['dim_x'], cfg['dim_y'], cfg['shared_dim'], cfg['unique_x_dim'],
                           cfg['unique_y_dim'], cfg['hidden_dim'], cfg['use_linear_decoder']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.scaler_x, model.scaler_y = ckpt['scaler_x'], ckpt['scaler_y']; model.eval()

    n = min(n, feats[pair[0]].shape[0])
    idx = np.random.choice(feats[pair[0]].shape[0], n, replace=False)
    X = model.scaler_x.transform(feats[pair[0]][idx]).astype(np.float32)
    Y = model.scaler_y.transform(feats[pair[1]][idx]).astype(np.float32)

    with torch.no_grad():
        Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
        sx, sy = model.shared_encoder_x(Xt).cpu().numpy(), model.shared_encoder_y(Yt).cpu().numpy()
        ux, uy = model.unique_encoder_x(Xt).cpu().numpy(), model.unique_encoder_y(Yt).cpu().numpy()

        # 获取 decoder 权重 (如果是线性的)
        has_linear_decoder = cfg.get('use_linear_decoder', False)
        if has_linear_decoder:
            # 尝试获取 unique decoder 权重
            try:
                Wu_x = model.unique_decoder_x[0].weight.cpu().numpy()  # (out, unique_dim)
                Wu_y = model.unique_decoder_y[0].weight.cpu().numpy()
            except:
                Wu_x, Wu_y = None, None
        else:
            Wu_x, Wu_y = None, None

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # === (a) Normalized Distance: ||sx - sy|| / (||sx|| + ||sy||) ===
    sx_norm = np.linalg.norm(sx, axis=1)
    sy_norm = np.linalg.norm(sy, axis=1)
    disp = np.linalg.norm(sx - sy, axis=1)
    norm_disp = disp / (sx_norm + sy_norm + 1e-8)

    axes[0,0].hist(norm_disp, bins=50, color=COLORS[pair[0]], alpha=0.7, edgecolor='white')
    axes[0,0].axvline(norm_disp.mean(), color='red', ls='--', lw=2, label=f'Mean: {norm_disp.mean():.3f}')
    axes[0,0].axvline(np.median(norm_disp), color='blue', ls=':', lw=2, label=f'Median: {np.median(norm_disp):.3f}')
    axes[0,0].set_xlabel('Normalized Distance: ||s_x - s_y|| / (||s_x|| + ||s_y||)', fontsize=10)
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('(a) Shared Code Alignment (Normalized)\n0 = Perfect, 1 = Orthogonal', fontsize=11, fontweight='bold')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)

    # === (b) Cosine Similarity ===
    sxn = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-8)
    syn = sy / (np.linalg.norm(sy, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(sxn * syn, axis=1)
    axes[0,1].hist(cos, bins=50, color=COLORS[pair[1]], alpha=0.7, edgecolor='white')
    axes[0,1].axvline(cos.mean(), color='red', ls='--', lw=2, label=f'Mean: {cos.mean():.3f}')
    axes[0,1].axvline(np.median(cos), color='blue', ls=':', lw=2, label=f'Median: {np.median(cos):.3f}')
    axes[0,1].set_xlabel('Cosine(shared_x, shared_y)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('(b) Shared Code Similarity\n(Higher = Better)', fontsize=11, fontweight='bold')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)

    # === (c) Unique Energy + Output Contribution ===
    ax_c = axes[1,0]
    uxe, uye = np.linalg.norm(ux, axis=1), np.linalg.norm(uy, axis=1)

    # 左: Unique norm histogram
    ax_c.hist(uxe, bins=40, alpha=0.5, color=COLORS[pair[0]], label=f'{pair[0]} ||u|| (μ={uxe.mean():.2f})')
    ax_c.hist(uye, bins=40, alpha=0.5, color=COLORS[pair[1]], label=f'{pair[1]} ||u|| (μ={uye.mean():.2f})')

    # 如果有线性 decoder, 计算输出贡献
    if Wu_x is not None and Wu_y is not None:
        # ||W_u @ u|| 作为输出贡献 proxy
        contrib_x = np.linalg.norm(ux @ Wu_x.T, axis=1)
        contrib_y = np.linalg.norm(uy @ Wu_y.T, axis=1)

        # 创建 twin axis
        ax_c2 = ax_c.twinx()
        ax_c2.hist(contrib_x, bins=40, alpha=0.3, color=COLORS[pair[0]],
                   histtype='step', lw=2, ls='--', label=f'{pair[0]} ||W·u|| (μ={contrib_x.mean():.2f})')
        ax_c2.hist(contrib_y, bins=40, alpha=0.3, color=COLORS[pair[1]],
                   histtype='step', lw=2, ls='--', label=f'{pair[1]} ||W·u|| (μ={contrib_y.mean():.2f})')
        ax_c2.set_ylabel('Output Contribution ||W·u||', fontsize=10)
        ax_c2.legend(fontsize=8, loc='upper right')

        ax_c.set_title('(c) Unique Code: Energy (filled) + Output Contribution (dashed)\n'
                      '||W·u|| ≈ Unique contribution to reconstruction', fontsize=10, fontweight='bold')
    else:
        ax_c.set_title('(c) Unique Code Energy\n(Higher = More Model-Specific)', fontsize=11, fontweight='bold')

    ax_c.set_xlabel('L2 Norm')
    ax_c.set_ylabel('Count')
    ax_c.legend(fontsize=9, loc='upper left')
    ax_c.grid(True, alpha=0.3)

    # === (d) ROC Curve + Effect Size ===
    rand_cos = np.sum(sxn * syn[np.random.permutation(n)], axis=1)

    # 计算 ROC
    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = np.concatenate([cos, rand_cos])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    au = roc_auc_score(y_true, y_score)
    cd = cohens_d(cos, rand_cos)
    delta_mean = cos.mean() - rand_cos.mean()

    axes[1,1].plot(fpr, tpr, color=COLORS[pair[0]], lw=2, label=f'ROC (AUC={au:.3f})')
    axes[1,1].plot([0, 1], [0, 1], color='gray', ls='--', lw=1, label='Chance')
    axes[1,1].fill_between(fpr, tpr, alpha=0.2, color=COLORS[pair[0]])

    # Effect size 文本框
    textstr = f"AUROC: {au:.3f}\nCohen's d: {cd:.2f}\nΔμ: {delta_mean:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1,1].text(0.6, 0.25, textstr, transform=axes[1,1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)

    axes[1,1].set_xlabel('False Positive Rate', fontsize=10)
    axes[1,1].set_ylabel('True Positive Rate', fontsize=10)
    axes[1,1].set_title('(d) ROC: Matched vs Random Shared Codes', fontsize=11, fontweight='bold')
    axes[1,1].legend(fontsize=10, loc='lower right')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(-0.02, 1.02)
    axes[1,1].set_ylim(-0.02, 1.02)

    fig.suptitle(f'SAE Decomposition: {pair[0]} vs {pair[1]}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize archived multi-model representations.")
    parser.add_argument("--cache-dir", required=True, help="Directory of aligned multi-model NPZs.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    out = args.output
    cache = os.path.join(args.cache_dir, "all_models_aligned_cache.npz")
    models = ["VLM", "ViT-Large", "ResNet101", "EVA-CLIP"]
    ref = "ResNet101"

    print("="*80 + "\n多模型表征空间高级可视化 v2 (优化版)\n" + "="*80)

    if not os.path.exists(cache): print(f"错误: 请先生成缓存"); return
    print(f"\n从缓存加载: {cache}")
    data = np.load(cache, allow_pickle=True)
    feats = {m: data[m] for m in models}
    print(f"已加载 {feats[models[0]].shape[0]} 个场景")

    print("\n计算 CKA...")
    cka = np.eye(4)
    for i in range(4):
        for j in range(i+1, 4):
            cka[i,j] = cka[j,i] = linear_cka(feats[models[i]], feats[models[j]])
    print("\nCKA:")
    for i, m in enumerate(models): print(f"  {m:<12}" + "".join([f"{cka[i,j]:>8.4f}" for j in range(4)]))

    plot_procrustes_residual(feats, models, os.path.join(out, "procrustes_v3.png"), ref)
    # plot_cka_network(cka, models, os.path.join(out, "cka_network_v2.png"))
    # plot_auroc(feats, models, os.path.join(out, "auroc_v2.png"), ref)
    # plot_recall(feats, models, os.path.join(out, "recall_v2.png"), ref)
    # plot_sae(feats, out, os.path.join(out, "sae_v2.png"))

    print("\n" + "="*80)
    print("Saved procrustes_v3.png")
    print(f"保存在: {out}")

if __name__ == "__main__":
    main()
