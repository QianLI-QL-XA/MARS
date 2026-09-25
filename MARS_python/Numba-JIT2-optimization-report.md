# MARS Python 版 JIT 化（第二阶段）报告

## 1. 背景

上一轮 profile（cProfile，control λ1 单点 3.5s）显示：Numba 第一版（v1）只 JIT 了**核心算子**，中高 λ 时
时间大头是 Python 解释层：`mars_path` 主循环 1.52s（38%）、`proxBmain` 包装 0.55s、`findmaxlambda` 0.37s、
numba 编译/缓存加载 0.62s。JIT 算子实际计算只占 ~12%。

## 2. 新增 6 个 JIT 内核（累计 11 个）

| 内核 | 替代 | 数学依据 |
|---|---|---|
| `_proxBmain_jit` | proxBmain 的 np.clip+triu | 对称 clamp，只扫上三角 |
| `_nonzero_upper_jit` | — | 提取 Omega 上三角非零 (i,j,v) |
| `_gradP_jit` | gradP 的稠密 matmul | 结合律：0.5·((Om@A)@A.T + A@(A.T@Om)) = 0.5·(Om@G + G@Om)，G=A@A.T 预计算一次；O(nnz×p) 稀疏感知 |
| `_tri_flat_gt_jit` | np.abs(triu)+flatnonzero | 列主序 flat 索引，等价 C++/Armadillo find() |
| `_maxlam_fromG_jit` | findmaxlambda 分块乘 | 复用完整 Gram G 直接扫上三角 |
| `_norm_sym_jit` | np.linalg.norm ×3 | 对称矩阵 Frobenius 范数只用上三角 |

所有内核 `@njit(cache=True)`，`USE_NUMBA=False` 自动回退纯 numpy（路径完全不变）。

## 3. 数值等价验证

| 检查 | 结果 |
|---|---|
| gradP JIT vs 原稠密式 | max|d| = 5.7e-14 |
| proxBmain JIT vs numpy | 0.0 |
| tri 扫描 JIT vs numpy | 完全一致 |
| maxlam G-scan vs 分块 | rel diff 1.4e-16 |
| 对称 norm | rel diff 2.2e-15 |
| 单点 obj（control λ1） | -3017.4815770 / s_off=68，位级一致 |

## 4. 真实数据性能（prostmat，3λ 路径，顺序运行）

| | R + MKL | numba v1 | numba v3 | v3 提升 |
|---|---:|---:|---:|---:|
| control（n=50） | 7.27 s | 11.32 s | **9.24 s** | 1.23× |
| cancer（n=52） | 9.26 s | 12.04 s | **10.87 s** | 1.11× |

分点（control/cancer）：v3 = 2.56/2.84/3.03、2.58/2.78/4.72；v1 = 3.68/3.42/3.59、3.21/3.16/5.02。

解一致性：中高 λ obj 与 v1 位级一致（cancer λ2 差 1.6e-6 系多解区边界浮点分支，s_off 相同）；
R vs Py 最大相对误差仍为 3.4×10⁻⁶。

## 5. 结论

- JIT 化第二阶段将 Python 总耗时压缩 10–18%，R(MKL) 仍快 1.17–1.27×。
- 剩余差距来自 **adaptive sieving 主循环与 Index/sub 管理的 Python 解释层**（np.unique、fancy indexing、
  数组组装）——整体 JIT 需重写外层循环，风险高、收益不确定。
- 单点视角的一次性开销（numba 缓存加载 ~0.6s、G=A@A.T 预计算 ~0.1s、新内核首次编译）在多 λ 路径下被摊薄。
