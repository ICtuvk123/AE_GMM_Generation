# AE + GMM / VBGMM 手写数字生成

用 **Autoencoder 降维 + GMM / VBGMM** 对 MNIST 建模并生成手写数字图像。

## 目录结构

```
AE_GMM_Generation/
├── data_loader.py   # MNIST 下载 & 加载
├── autoencoder.py   # MLP Autoencoder
├── gmm.py           # 手写 GMM + EM（full / diag / spherical）
├── vbgmm.py         # 手写 VBGMM（自动裁剪成分）
├── representation.py# AE 表示层接口
├── train.py         # 训练入口
├── generate.py      # 采样生成图像
├── evaluate.py      # 评估：K sweep / t-SNE
└── requirements.txt
```

## 环境准备

```bash
pip install -r requirements.txt
```

第一次运行 `train.py` 或 `evaluate.py` 时，程序会自动下载 MNIST 到 `data/` 目录。

## 方法概述

```
┌─────────────────────────────────────────────────────────────┐
│                    两阶段生成流程                            │
├─────────────────────────────────────────────────────────────┤
│  阶段 1: 表示学习 (Autoencoder)                              │
│  ┌─────────────┐                                            │
│  │ Autoencoder │  784 → 512 → 256 → latent_dim → 256 → 512 → 784  │
│  └──────┬──────┘                                            │
│         ↓ 提取 latent vector                                 │
├─────────────────────────────────────────────────────────────┤
│  阶段 2: 概率密度建模 (GMM/VBGMM)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GMM (EM 算法) 或  VBGMM (变分贝叶斯)                 │    │
│  │  - 学习 latent space 的多峰分布                        │    │
│  │  - 支持自动成分裁剪 (VBGMM)                           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  生成：采样 z ~ GMM → Decoder → 图像                          │
└─────────────────────────────────────────────────────────────┘
```

## 训练说明

项目的训练入口统一是：

```bash
python train.py [AE 参数] [混合模型参数] [训练模式参数]
```

训练时主要有 2 个维度可选：

- 混合模型：`--model gmm` 或 `--model vbgmm`
- 训练模式：`--mode perclass` 或 `--mode global`

推荐配置：

- `perclass`：每个数字训练一个独立模型，便于生成指定类别
- `diag` 协方差：训练快、稳定性好

### 1. 训练 AE + GMM

训练脚本会自动先训练自编码器并保存，然后训练 GMM。

```bash
python train.py \
  --ae_latent_dim 32 \
  --ae_epochs 30 \
  --ae_batch_size 256 \
  --ae_lr 1e-3 \
  --model gmm \
  --n_components 10 \
  --cov_type diag \
  --mode perclass \
  --max_iter 100
```

这会同时产出：

```bash
checkpoints/autoencoder_ld32.pt
checkpoints/model_perclass_ae_gmm_k10.pkl
```

### 2. 复用已训练的 AE

如果 AE 已经训练好了，后续做多 `K` 对比时建议直接复用已有 AE checkpoint：

```bash
python train.py \
  --ae_ckpt checkpoints/autoencoder_ld32.pt \
  --model gmm \
  --n_components 10 \
  --cov_type diag \
  --mode perclass \
  --max_iter 100
```

多 `K` 实验时只需要改 `--n_components`：

```bash
python train.py --ae_ckpt checkpoints/autoencoder_ld32.pt --model gmm --n_components 2  --cov_type diag --mode perclass --max_iter 60
python train.py --ae_ckpt checkpoints/autoencoder_ld32.pt --model gmm --n_components 10 --cov_type diag --mode perclass --max_iter 60
python train.py --ae_ckpt checkpoints/autoencoder_ld32.pt --model gmm --n_components 20 --cov_type diag --mode perclass --max_iter 60
```

对应会生成：

```bash
checkpoints/model_perclass_ae_gmm_k2.pkl
checkpoints/model_perclass_ae_gmm_k10.pkl
checkpoints/model_perclass_ae_gmm_k20.pkl
```

### 3. 训练 AE + VBGMM

这是项目的重点推荐方案，因为它结合了：

- Autoencoder 的非线性表示学习
- VBGMM 的自动成分裁剪能力

```bash
python train.py \
  --ae_latent_dim 32 \
  --ae_epochs 30 \
  --model vbgmm \
  --n_components 50 \
  --mode perclass \
  --max_iter 300 \
  --alpha_0 0.01
```

如果已有 AE，也可以复用：

```bash
python train.py \
  --ae_ckpt checkpoints/autoencoder_ld32.pt \
  --model vbgmm \
  --n_components 50 \
  --mode perclass \
  --max_iter 300 \
  --alpha_0 0.01
```

训练完成后会生成类似文件：

```bash
checkpoints/model_perclass_ae_vbgmm_k50.pkl
```

## 生成图像

训练完成后，统一用 `generate.py` 采样。

### 1. 生成单个实验的结果

```bash
python generate.py --ckpt checkpoints/model_perclass_ae_gmm_k10.pkl  --n 64 --show_prototypes
python generate.py --ckpt checkpoints/model_perclass_ae_vbgmm_k50.pkl --n 64 --show_prototypes
```

### 2. 给不同实验结果加前缀，避免覆盖

推荐在做对比实验时加 `--tag`：

```bash
python generate.py --ckpt checkpoints/model_perclass_ae_gmm_k10.pkl  --tag ae_gmm_k10  --n 64 --show_prototypes
python generate.py --ckpt checkpoints/model_perclass_ae_vbgmm_k50.pkl --tag ae_vbgmm_k50 --n 64 --show_prototypes
```

这样输出会保存成：

```bash
outputs/ae_gmm_k10_all_digits.png
outputs/ae_vbgmm_k50_all_digits.png
```

## 评估

### 1. GMM 的 K 扫描

```bash
python evaluate.py --k_sweep --digit 3
```

### 2. 对训练好的模型做测试集评估

```bash
python evaluate.py --ckpt checkpoints/model_perclass_ae_gmm_k10.pkl
python evaluate.py --ckpt checkpoints/model_perclass_ae_vbgmm_k50.pkl
```

### 3. t-SNE 可视化

```bash
python evaluate.py --ckpt checkpoints/model_perclass_ae_vbgmm_k50.pkl --tsne --digit 3
```

## 主要参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--ae_latent_dim` | AE 潜变量维度 | 16 ~ 64 |
| `--ae_epochs` | AE 训练轮数 | 30 ~ 50 |
| `--ae_batch_size` | AE 训练 batch size | 256 |
| `--ae_lr` | AE 学习率 | 1e-3 |
| `--ae_ckpt` | 预训练 AE 路径（可选） | - |
| `--n_components` | 每类混合成分数或 VBGMM 最大成分数 | `10 ~ 20`（GMM）`30 ~ 50`（VBGMM） |
| `--cov_type` | 协方差类型，仅对 GMM 生效 | `diag`（推荐） |
| `--mode` | 训练模式 | `perclass` / `global` |
| `--model` | 混合模型类型 | `gmm` / `vbgmm` |
| `--alpha_0` | VBGMM 的 Dirichlet 先验浓度 | `0.01 ~ 1/K` |
| `--max_iter` | 混合模型最大迭代次数 | `60 ~ 300` |

## 推荐训练路线

1. `AE + GMM (K=10)`：基础配置，快速验证流程
2. `AE + GMM (K=2, 10, 20)`：对比不同 K 的生成效果
3. `AE + VBGMM (K_max=50)`：作为最终方案，观察自动成分裁剪

## 核心算法说明

### EM 算法流程

```
初始化：K-means++ 初始化均值，全局方差初始化协方差

循环直到收敛：
  E-step：计算每个样本属于各组件的后验概率 r_{nk}
          r_{nk} ∝ π_k · N(x_n | μ_k, Σ_k)
  M-step：用加权 MLE 更新参数
          π_k = N_k / N
          μ_k = Σ_n r_{nk} x_n / N_k
          Σ_k = Σ_n r_{nk} (x_n - μ_k)(x_n - μ_k)^T / N_k
```

### 生成流程

```
采样 k ~ Categorical(π)
采样 z ~ N(μ_k, Σ_k)
解码 x = Decoder(z) → 28×28 图像
```

## 方法局限性与改进方向

### 当前方法的局限

1. **两阶段解耦训练**：AE 和 GMM 分别优化，没有联合目标函数
2. **Latent 分布匹配问题**：GMM 拟合的分布与 AE 编码的实际分布可能存在差异
3. **生成质量不稳定**：GMM 采样可能落在 Decoder 的"流形外"区域，导致生成模糊

### 改进方向

- **联合优化**：交替训练 AE 和 GMM，使 latent 分布更接近高斯混合
- **VAE 方案**：使用 VAE 直接约束 latent 分布为标准正态
- **后处理增强**：使用 `--sharpen` 参数增强生成图像的对比度
