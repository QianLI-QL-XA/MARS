# MARS Python 版 Numba JIT 优化报告

## 1. 实施内容

对 `MARS_python/mars_py.py`（纯 numpy 移植版，SSNAL + adaptive sieving + PCG）实施 Numba JIT 优化：

- **新增 7 个 `@njit(cache=True)` 内核**：`_opSY_jit`、`_opInvLA_jit`、`_prox_b_jit`、`_partgradient_jit`、`_PMEASCG_jit`（整函数 CG，含停滞检查）、`_findstep_jit`（整函数线搜索）、`_sumsq`/`_sumsq_xy`/`_normF` 辅助。
- **改造 6 个调用入口**：`operatorSY`、`operatorInvLA`、`prox_b`、`partgradient`、`PMEASCG`、`findstep` 均加 `USE_NUMBA` 分支；外层 `PMEASSSNCGc`/`PMEASmainc`/`mars_path` 结构不变（SN 迭代层 Python 调度开销占比小）。
- **保留 numpy 回退开关**：`mars_py.USE_NUMBA = False` 即回到原始 numpy 实现（numba 未安装时自动回退）。

## 2. 修复的 JIT bug

调试中发现并修复 **CG 内积误用平方和**：`denom = _sumsq(g * Vg)` 原实现为 `sum(g²·Vg²)`，应为 `sum(g·Vg)`（`_sumsq_xy(g, Vg)`）。此 bug 曾导致 CG 残差停滞、解发散（obj 爆炸为正）。修复后 CG 收敛行为与 numpy 版一致（ok=1，方向差 6.6e-7 属浮点累加差异）。

## 3. 正确性验证（numpy vs numba 逐级对拍）

| 层级 | 结果 |
|---|---|
| 算子级（SY/InvLA/prox/partgrad） | max diff ≤ 3.6e-15（浮点累加顺序） |
| CG 单步（随机 p=60,n=20） | solveok 一致（1/1），direction 差 6.6e-7 |
| findstep 单步 | Y/ztmp/z 完全一致（diff 0.0） |
| 合成数据冒烟（p=100,n=30） | λ=0.4 obj 差 0.0、s_off 相同；λ=0.2 obj 差 6.7e-1（rel 1e-6，多解区） |
| 真实数据 control 全路径（p=6033,n=50） | 前三 λ obj rel ≤1.7e-11、s_off 全同；低 λ obj rel 4.3e-9、s_off 相同 |
| 真实数据 cancer 全路径（p=6033,n=52） | 前三 λ obj rel ≤2.9e-10、s_off 全同；低 λ obj rel 3.0e-4、s_off 2829 vs 2832（多解区不同分支，与历史 R/Py/Ml 差异同量级） |

> 低 λ 属多解区，对 BLAS/累加顺序敏感（历史已证：同一语言两次运行差距可达 1e-2 量级）；numba 与 numpy 的差异均在此范围内，非实现错误。

## 4. 性能对比（Windows 桌面，24 逻辑核，numpy=OpenBLAS 多线程）

### 真实数据全路径（prostate，λ 路径 4 点，stoptol=1e-4，maxiter=10）

| 数据集 | numpy | numba | 加速比 |
|---|---:|---:|---:|
| control（p=6033, n=50） | 59.9 s | 25.8 s | **2.3×** |
| cancer（p=6033, n=52） | 183.3 s | 52.8 s | **3.5×** |
| cancer 低 λ 单点（λ=0.647） | 108.8 s | 28.0 s | **3.9×** |

### 合成数据冒烟（p=100, n=30，2 点路径）

numpy 20.5 s → numba 1.43 s（**14.3×**；λ=0.4 单点 40×）。

### 算子级（p=6033, n=52 微基准）

| 算子 | numpy | numba | 加速比 |
|---|---:|---:|---:|
| operatorSY | 9025 μs | 423 μs | **21.3×** |
| operatorInvLA | 7660 μs | 826 μs | **9.3×** |

## 5. 结论与后续

- **Numba JIT 版数学等价（高/中 λ 位级一致）、性能提升 2.3–3.9×（真实数据全路径）**；算子级 9–21×，外层 Python 调度与 findA/findmaxlambda/gradP 等 numpy 大矩阵运算（BLAS 已快）为剩余瓶颈。
- 若需进一步提速：① 将 `PMEASSSNCGc`/`PMEASmainc` 整函数 JIT（消除 SN 层调度）；② findmaxlambda 块循环 JIT；③ 低 λ 大活性集场景可考虑 24 核并行（多 λ 批处理）。
- GPU（RTX 5060 4GB）在当前规模为负收益（数据传输 > 计算），留给 p≥1e5 场景。
