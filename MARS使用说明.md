# MARS 完整使用说明（R / Python）

MARS（**M**ARS: **A** **second-order reduction algorithm for high-dimensional
**s**parse precision matrices estimation）是估计**稀疏精度矩阵**（precision
matrix，即协方差矩阵的逆 `Omega = Sigma^{-1}`）的高维算法包，同时提供 R 与
Python 两个实现。本文档覆盖：问题背景、模型与算法、R 包安装与使用、
Python 包安装与使用、真实数据示例、性能调优与常见问题。

---

## 1. 问题背景：精度矩阵与条件相关

给定 `n` 个 i.i.d. 样本（列）组成的 `p x n` 数据矩阵 `X`（`p` 为变量数，
`n` 为样本量，高维场景下 `p >> n`）。目标是估计精度矩阵 `Omega`。

**为什么关注精度矩阵？** 对高斯图模型（Gaussian graphical model）：

- 精度矩阵的 (i, j) 元素 `Omega_ij = 0` **当且仅当** 变量 i 与 j 在给定
  其余所有变量的条件下**条件独立**（conditional independence）；
- 因此 `Omega` 的非零模式定义了一张**图**：节点 = 变量，边 = 非零的
  `Omega_ij`，表示两个变量之间的**条件相关**（partial correlation，
  经对角元素归一化后即条件相关系数）。

MARS 通过加 `l1` 惩罚的 D-trace loss 在 `p >> n` 下得到**稀疏**的精度矩阵
估计，即稀疏的图——在高维生物医学数据（如基因表达）中，这意味着只保留
统计上显著的直接关联。

## 2. 模型与算法

求解如下优化问题（`Omega` 对称）：

```
min_{Omega in S^p}  0.5 || Omega A ||_F^2 - <Omega, I> + lam || Omega ||_{1, off}
```

- `A`：由样本中心化 + thin SVD 得到的变换矩阵（`A = U diag(s)/sqrt(n-1)`）；
- `||Omega||_{1, off} = sum_{i != j} |Omega_ij|`：非对角元的 l1 惩罚，诱导稀疏图；
- `lam`：调优参数，控制稀疏度（越大越稀疏）。

算法 = **自适应筛选（adaptive sieving, AS）** 保持活跃集（active set）小 +
**半光滑牛顿增广拉格朗日（SSNAL）** 求解子问题 + **共轭梯度（PCG）** 内层
线性系统。`p` 很大时不会显式构造 `p x p` 的 Gram 矩阵。

## 3. R 包使用

### 3.1 安装

依赖：R (>= 3.5) + C++ 工具链（Windows 装 Rtools，macOS 装 Xcode CLT，
Linux 装 g++）+ 包 `Rcpp`、`RcppArmadillo`、`Matrix`。

```r
install.packages(c("Rcpp", "RcppArmadillo", "Matrix"))

# 本地安装（在包含 DESCRIPTION 的目录）
install.packages(".", repos = NULL, type = "source")
# 或 GitHub
# library(devtools); devtools::install_github("QianLI-QL/MARS")
```

### 3.2 核心函数 `MARS()`

```r
MARS(X, Lambdapath, stopmethod = c("bigs", "fix"), fixnumber = 1L,
     maxiter = 100L, maxlambdacheck = TRUE, printmain = FALSE,
     printsub = FALSE, stoptol = 1e-4, sigma = 1.0)
```

| 参数 | 说明 |
|---|---|
| `X` | `p x n` 样本矩阵（行 = 变量，列 = 样本） |
| `Lambdapath` | 候选调优参数（递减）。`maxlambdacheck=TRUE` 时自动丢弃超过数据依赖 max-lambda 的值 |
| `stopmethod` | `"bigs"`：活跃集饱和时提前停（`nnz > length(Lambdapath)*n` 或 `numAS > 4`）；`"fix"`：沿路径逐点求解 |
| `fixnumber` | 配合 `"fix"`：求解前几个 lambda |
| `maxiter` | 外层（ALM）最大迭代 |
| `maxlambdacheck` | 是否过滤超过 max-lambda 的调优参数 |
| `printmain` / `printsub` | 打印主/子问题进度 |
| `stoptol` | 停机容差；满足 `max(primfeas, dualfeas) < 500*max(1e-6, stoptol)` **且** `eta < stoptol`（含 `gap < tol` 判据）时停止 |
| `sigma` | ALM 惩罚参数初值 |

**返回值**：列表
- `Omegapath`：每个 lambda 对应的稀疏精度矩阵（`dsCMatrix`，Matrix 包稀疏对称格式）；
- `Lambdapath`：实际使用的 lambda；
- `timepath`：每个 lambda 的耗时（秒）。

### 3.3 对比求解器 `PMEAS()`

```r
PMEAS(X, Lambdapath, calmethod = c("SSNAL", "iADMM", "eADMM"), ...)
```

同一接口下提供三种求解器：默认 SSNAL、不精确 ADMM（`"iADMM"`）、
精确 ADMM（`"eADMM"`），用于交叉验证解与基准对比。

### 3.4 真实数据示例（前列腺基因表达）

```r
library(MARS)
prost <- read.csv("prostmat.csv", header = FALSE)      # 6033 x 102
X1 <- as.matrix(prost[, 1:50])                          # control，n = 50
X2 <- as.matrix(prost[, 51:102])                        # prostate cancer，n = 52

sol1 <- MARS(X1, Lambdapath = c(0.95, 0.85, 0.75, 0.65) * maxLambda(X1),
             stopmethod = "fix", maxiter = 10, stoptol = 1e-4)
sol2 <- MARS(X2, Lambdapath = c(0.95, 0.85, 0.75, 0.65) * maxLambda(X2),
             stopmethod = "fix", maxiter = 10, stoptol = 1e-4)

# 非零模式即基因网络：Omega_ij != 0 表示基因 i、j 条件相关
Omega <- as.matrix(sol1$Omegapath[[4]])
plot(which(Omega != 0 & upper.tri(Omega), arr.ind = TRUE))   # 简易网络示意
```

## 4. Python 包使用

### 4.1 安装

```bash
cd MARS_python
pip install -e .            # 核心（仅 NumPy）
pip install -e ".[jit]"     # + Numba JIT 内核（推荐）
```

未安装 Numba 时自动回退纯 NumPy（`USE_NUMBA = False`）。

### 4.2 快速上手

```python
import numpy as np
from mars import mars_path, maxLambda

rng = np.random.default_rng(0)
X = rng.standard_normal((3000, 50))              # p = 3000, n = 50
lam = 0.95 * maxLambda(X)                        # 数据依赖 max-lambda

Omegapath, Lambdapath, timepath = mars_path(
    X, np.array([lam]), stoptol=1e-4, maxiter=10, stopmethod="fix")
Omega = Omegapath[0]                             # 3000 x 3000 对称矩阵
```

### 4.3 API

| 函数 | 说明 |
|---|---|
| `mars_path(X, Lambdapath, stoptol=1e-4, maxiter=10, stopmethod="fix", printyes=False, printyessub=False, sigma=1.0, maxlambdacheck=True)` | 主入口：求解整条 lambda 路径；返回 `(Omegapath, Lambdapath, timepath)` |
| `findA(X)` | 构建变换矩阵 A（行中心化 + thin SVD） |
| `maxLambda(X)` / `findmaxlambda(A)` | 数据依赖 max-lambda（不构造 p×p Gram） |
| `PMEASmainc(A, lam, ...)` | 在给定活跃集上求解单个 lambda（底层） |
| `operatorSY` / `operatorInvLA` / `prox_b` / `proxBmain` / `partgradient` / `PMEASCG` / `findstep` / `PMEASSSNCGc` | C++ 内核对等移植，供实验与调试 |

参数语义与 R 版一致（见上表，仅 `printmain`→`printyes` 命名差异）。

### 4.4 数值等价性

Python 与 R 在算子级逐迭代对拍一致（机器精度内）；低 lambda 多解区可能因
BLAS 线程差异落入不同最优分支——该区域请**比较目标函数值（绝对/相对误差）**
而非逐元素解。

## 5. 性能调优

- **R**：换多线程 BLAS（如 Intel MKL）可将矩阵密集段提速 5-6 倍
  （Windows 实操见 `MARS_matlab/MARS-R-MKL-optimization.md`）。
- **Python**：启用 Numba JIT（11 个内核）比纯 NumPy 快 3-4 倍。
- 真实数据（prostmat，p=6033，3 点 lambda 路径，顺序）参考：

| 实现 | control (n=50) | cancer (n=52) |
|---|---:|---:|
| R + Intel MKL | 7.27 s | 9.26 s |
| Python + Numba | 9.24 s | 10.87 s |
| Python 纯 NumPy | ~14.8 s | ~16 s |

## 6. 常见问题

**Q: 输出矩阵里非零元素代表什么？**
非零 `Omega_ij`（i≠j）表示变量 i、j 在给定其余变量后仍条件相关；对角元素
归一化后得到偏相关系数。整张图的稀疏性由 lambda 控制。

**Q: 低 lambda 下 R 与 Python 的解为何略有不同？**
低 lambda 靠近病态多解区，解的**非唯一**导致极小浮点差被放大到不同最优
分支。判断正确性请比较目标函数值（相对误差通常 < 1e-3 量级属正常分支差）。

**Q: p 很大时内存是否爆？**
算法不构造 p×p Gram 矩阵；活跃集（AS）随 lambda 保持较小，内存 O(p·n)
级别。

**Q: 如何选 lambda？**
对无监督图模型估计，可用五折交叉验证选择使似然/预测损失最小的 lambda
（见论文第 5 节）。

## 7. 文件结构

```
MARS/
├── DESCRIPTION / NAMESPACE / man/ / src/   # R 包（MARSc.cpp 主内核）
├── R/MARS.R, R/PMEAS.R                      # R 接口
├── tests/                                   # R 冒烟测试
├── data/prostmat.csv                        # 真实数据（6033 x 102）
├── README.md                                # R 包英文文档
├── MARS_python/
│   ├── mars/ (__init__.py + mars_py.py)     # Python 包（pip install -e .）
│   ├── pyproject.toml
│   └── README.md
├── MARS_matlab/                             # MATLAB/Octave 移植与基准
└── MARS-perf-dashboard/                     # 性能对比网页
```

## 8. 引用

Qian LI, Binyan Jiang, and Defeng Sun. "MARS: A second-order reduction
algorithm for high-dimensional sparse precision matrices estimation."
*Journal of Machine Learning Research*, 24 (2023) 1-44.
