# MARS R 包代码审查与修复报告

审查对象：`C:\Users\qianl\OneDrive\codes\MARS`（MARS 论文随附 R 包源码）
审查依据：Qian Li, Binyan Jiang, Defeng Sun, *MARS: A Second-Order Reduction Algorithm for High-Dimensional Sparse Precision Matrices Estimation*, JMLR 24 (2023) 1-44（即本目录 MARS.pdf）

结论：**源码存在 11 处真实错误（其中 2 处致命）与 13 项可优化项，已全部修复/完成；此后又新增 4 项真实工具链修复（共 15 项错误修复 + 13 项优化）。最终版本 0.2.0 通过 `R CMD check --as-cran` 零提示（含 PDF 参考手册）。**
最致命的问题：原源码因 `#define DOUBLE_EPS 1E-15;`（宏尾带分号）**根本无法编译**，目录内旧 `.dll/.o` 是此前旧版本的残留产物。

---

## 一、致命错误（不修复则无法编译 / 无法运行）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `src/MARSc.cpp`、`src/PMEASc.cpp`、`src/MARSc.h`、`src/PMEASc.h` | `#define DOUBLE_EPS 1E-15;` 宏定义带尾分号。所有 `(DOUBLE_EPS)` 用法展开为 `(1E-15;)`，编译报错 `expected ')' before ';'`（已用 g++ 实证） | 去掉尾分号：`#define DOUBLE_EPS 1e-15`（4 处） |
| 2 | `NAMESPACE` | 缺少 `useDynLib(MARS, .registration = TRUE)`。R 加载包时不加载 DLL，`.Call('_MARS_MARSc', PACKAGE='MARS', ...)` 运行时报“DLL not found” | 重写 NAMESPACE，加入 `useDynLib(MARS, .registration = TRUE)` 与 `importFrom(Rcpp, evalCpp)`；新增 `R/MARS-package.R` 写入 `@useDynLib` 标记，保证日后重跑 roxygen2 不会再次丢失 |

## 二、R 层输入校验与默认参数错误

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 3 | `R/MARS.R`、`R/PMEAS.R` | `match(stopmethod, ...) == 0`：非法输入时 `match` 返回 `NA`，`if(NA)` 直接崩溃且不显示错误信息 | 改用 `%in%` 判定，非法输入给出明确报错 |
| 4 | `R/MARS.R`、`R/PMEAS.R` | `is.integer(fixnumber)` 拒绝默认值 `fixnumber = 10`（R 里是 double），**默认调用即报错** | 改为“正整数”判定（`fixnumber == floor(fixnumber)`），通过后 `as.integer()` 转换 |
| 5 | `R/PMEAS.R` | 默认 `calmethod = c("SSNAL","iADMM","eADMM")` 是三元素向量，`match()` 返回长度 3 向量，`if()` 报 “condition has length > 1”——**默认调用即报错** | 默认值改为 `"SSNAL"`，并用 `%in%` 校验 |
| 6 | `R/MARS.R`、`R/PMEAS.R` | `sigma` 允许 0（提示却说“正数”）；子问题中含 `x/sigma`，sigma=0 会除零 | 校验收紧为 `sigma > 0` |
| 7 | `R/MARS.R` | 错误提示文案把参数名写成 `stoptol`（应为 `stopmethod`），易误导 | 修正文案 |

## 三、运行 / 数值安全错误

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 8 | `src/MARSc.cpp` | 自适应筛选中 `sortindex(span(0, nonzeronumberIndex))` 存在 off-by-one 且活动集很大时可能越界崩溃 | 用 `min(nonzeronumberIndex, n_elem-1)` 截断，且为 0 时走兜底分支 |
| 9 | `src/MARSc.cpp`、`src/PMEASc.cpp` | `maxlambdacheck=TRUE` 且所有 λ ≥ maxλ 时路径被清空，原代码静默返回空结果 | 增加空路径保护：`Rf_warning` 提示并返回空列表 |

## 四、性能优化（数值结果不变）

| # | 文件 | 优化 |
|---|------|------|
| 10 | `src/MARSc.cpp` | `findmaxlambda` 原来构造完整 p×p 矩阵 `S = A*A.t()`（p=22283 时约 4GB），改为利用 `S(i,j)=<A_i,A_j>` 逐项计算，峰值内存由 O(p²) 降到 O(n)——这正是论文主打的“高维省内存”场景 |
| 11 | `src/PMEASc.cpp` | SSNAL/iADMM 路径不再构造完整 `S`（仅 eADMM 需要），对角元与 maxλ 就地计算，峰值内存减半 |
| 12 | `src/MARSc.cpp` | 主循环 `gradP` 去掉每次分配 `eye(p,p)`，改为 `gradP.diag() -= 1.0` |

## 五、代码质量

| # | 文件 | 处理 |
|---|------|------|
| 13 | `src/MARSc.cpp` | `double timepath` 遮蔽外层 `arma::vec timepath`，改为 `timepath_total` |
| 14 | `src/MARSc.cpp` | 返回 `Omegapath` 时显式逐元素转 `Rcpp::List`，消除 `field<sp_mat>` 包装的不确定性 |
| 15 | `src/MARSc.cpp`、`src/PMEASc.cpp` | 删除死代码：`findnewJ`、`itersub > 1000` 永不执行块（maxitersub ≤ 30）、`vectoraccess` |
| 16 | `src/MARSc.h` | 补 `#include <Rcpp.h>`（原头文件用 `Rcpp::List` 却不含 Rcpp 头）；修正头文件守卫名 |

## 六、包工程化

| # | 处理 |
|---|------|
| 17 | `NAMESPACE` 重写：只导出公共 API `MARS`、`PMEAS`；`MARSc`/`PMEASc` 转为内部包装（`RcppExports.R` 移除 `@export`，`src/MARSc.cpp` 移除 `//' @export`），避免“导出但无文档”的 check 告警 |
| 18 | `DESCRIPTION` 补齐：`Authors@R`、`URL`、`BugReports`、`NeedsCompilation: yes`，版本升至 0.2.0 |
| 19 | `man/MARS.Rd`、`man/PMEAS.Rd` 更新：修正默认值（PMEAS 的 calmethod）、补充 `\examples`、完整参考文献 |
| 20 | 新增 `tests/` 冒烟测试（`mars-smoke.R`、`pmeas-smoke.R`），断言返回结构、对称性、有限值与正对角 |
| 21 | 新增 `.Rbuildignore`（排除 .Rproj / .Rhistory / .pdf / 编译产物） |
| 22 | 清理旧构建残留：`src/*.o`、`src/MARS.dll`、`R/.Rhistory`、`src/.Rhistory`、顶层旧测试脚本 |
| 23 | `README.md` 更新（本地安装方式、返回值说明） |
| 24 | 新增 `R/MARS-package.R`（`@useDynLib` / `@importFrom Rcpp evalCpp`），防止日后重跑 roxygen2 丢失注册信息 |

---

## 七、真实工具链验证新增修复（R 4.6.1 + Rtools45 实测发现）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 25 | `src/MARSc.cpp` | R 4.6 默认 `-std=gnu++20`，GCC 14 的 `<span>` 头使裸 `span(...)` 在 `std::span` 与 `arma::span` 间**歧义**，编译报 `reference to 'span' is ambiguous`（旧 R 用 C++11 时不会触发） | 显式写 `arma::span(0, nkeep - 1)` |
| 26 | `src/MARSc.cpp`、`src/PMEASc.cpp` | `R CMD check` 检出 `std::cout`（`_ZSt4cout`）：编译代码不应直接写进程 stdout | 57 处 `cout` → `Rcpp::Rcout`（路由到 R 控制台，行为不变） |
| 27 | `NAMESPACE` / `R/MARS-package.R` | `Depends: Matrix` 未在 NAMESPACE 导入，产生 NOTE | 增加 `importClassesFrom(Matrix, dgCMatrix)`（返回的稀疏矩阵正是 Matrix 的 dgCMatrix 类） |
| 28 | `.Rbuildignore` | 审查报告 `MARS-review-report.md` 混入 tarball 顶层，产生 NOTE | 加入 `.Rbuildignore`（报告保留在项目内，不随包发布） |

## 验证情况（本机实测，2026-09-24，最终版）

- 环境：**Windows 11 + R 4.6.1（UCRT）+ Rtools45（gcc 14.3.0）+ TinyTeX（TeX Live 2026）+ pandoc 3.11**。Rtools45 在 `C:\rtools45`；TinyTeX 在 `C:\Users\qianl\AppData\Roaming\TinyTeX`；pandoc 3.11 在 `C:\Users\qianl\pandoc\pandoc-3.11`。
- `R CMD build --no-build-vignettes`：成功生成 **MARS_0.2.0.tar.gz**。
- `R CMD INSTALL --preclean`：C++ 在 `-O2 -Wall -std=gnu++20` 下**零警告编译**；`tests/mars-smoke.R`、`tests/pmeas-smoke.R` **全部通过**（MARS 的 SSNAL；PMEAS 的 SSNAL/iADMM/eADMM）。
- `R CMD check --as-cran MARS_0.2.0.tar.gz`：**Status: OK —— 0 ERROR / 0 WARNING / 0 NOTE**，54+ 项检查全部 OK，含：
  - `checking top-level files ... OK`（README 经 **pandoc** 校验通过，无 NOTE）；
  - `checking PDF version of manual ... OK`（**TinyTeX/pdflatex** 生成 4 页 PDF 手册，75.3 KB）；
  - `checking HTML version of manual ... OK`；tests 全过；无残留文件 NOTE。
- 过程中的技术点（记录备查）：
  - TinyTeX 安装后需补齐宏包：**courier**（`\url{}` 用 Courier 字体 pcrr8t.tfm，TinyTeX 默认缺）与 **makeindex**（索引工具，默认缺）；`tlmgr install courier makeindex` 解决。
  - pandoc 3.11 由 GitHub 官方 release zip 安装（winget 后台安装未落地）。
  - R CMD check 的 subprocess 继承调用方环境变量（TEXMFROOT/TEXMFCNF 有效），但 R 进程启动时会把 `R_HOME\bin\x64` 前置到自身 PATH——曾因把 pdflatex 拷贝进 `R_HOME/bin/x64` 导致 Sys.which 优先命中残缺副本而失败，**已全部移除拷贝、回归 TinyTeX 原生 pdflatex**。
  - CRAN 主站 incoming 网络检查在本机超时，设 `_R_CHECK_CRAN_INCOMING_REMOTE_=false` 跳过（环境网络限制，非包问题）。

### 构建与检查命令（已实测可用）

```r
# 安装（0.2.0）
# R CMD INSTALL --preclean C:\Users\qianl\OneDrive\codes\MARS   # 或
# install.packages("C:/Users/qianl/OneDrive/codes/MARS", repos = NULL, type = "source")

# 完整检查（含 tests/ 冒烟测试与 PDF 手册；需 pandoc 与 TinyTeX 在 PATH）
# $env:Path = "$env:APPDATA\TinyTeX\bin\windows;C:\Users\qianl\pandoc\pandoc-3.11;$env:Path"
# $env:TEXMFROOT = "$env:APPDATA\TinyTeX"; $env:TEXMFCNF = "$env:APPDATA\TinyTeX\texmf-dist\web2c"
# $env:_R_CHECK_CRAN_INCOMING_REMOTE_ = "false"
# R CMD build --no-build-vignettes C:\Users\qianl\OneDrive\codes\MARS
# R CMD check --as-cran MARS_0.2.0.tar.gz
```

### 说明与残余提示

- 主循环 KKT 残差仍按全 p×p 矩阵计算（论文算法本身如此），超大 p 场景请关注内存；若未来需要，可考虑启用 `ARMA_64BIT_WORD`（已注释保留）。
- `fixnumber` 现在对 `"bigs"`/`"fix"` 两种 stopmethod 都做正整数校验（C++ 中两种模式都会用到它）。
- 安装目录：`C:\Users\qianl\R\R-4.6.1`（R）、`C:\rtools45`（Rtools45）、`C:\Users\qianl\AppData\Roaming\TinyTeX`（TinyTeX）、`C:\Users\qianl\pandoc\pandoc-3.11`（pandoc）。
