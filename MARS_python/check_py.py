# sanity check: run pure-python MARS on small data, verify symmetry/finiteness/KKT
import numpy as np, sys, time
sys.path.insert(0, "C:/Users/qianl/OneDrive/codes/MARS/MARS_python")
from mars_py import mars_path, findA

def kkt(Om, X, lam):
    A = findA(X); S = A @ A.T; p = Om.shape[0]
    h = 0.5 * (S @ Om + Om @ S) - np.eye(p)
    Z = np.clip(-h / lam, -1, 1); np.fill_diagonal(Z, 0)
    return float(np.linalg.norm(h + lam * Z))

rng = np.random.default_rng(1)
X = rng.standard_normal((30, 10))
lams = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3])
t0 = time.perf_counter()
Op, Lp, tp = mars_path(X, lams, stoptol=1e-4, maxiter=10, printyes=False)
t1 = time.perf_counter()
print("python mars_path elapsed: %.3f s" % (t1 - t0))
print("lambda path:", Lp)
for k, Om in enumerate(Op):
    sym = np.allclose(Om, Om.T, atol=1e-8)
    fin = np.all(np.isfinite(Om))
    r = kkt(Om, X, Lp[k])
    nnz = int(np.sum(Om != 0) - Om.shape[0])
    print("lam=%.2f sym=%s finite=%s kkt=%.2e nnz_off=%d time=%.3f" % (Lp[k], sym, fin, r, nnz, tp[k]))
print("OK")
