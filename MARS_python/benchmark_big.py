# Large-dimension benchmark: p=1000 (dense compare), p=5000 (sparse compare), n=50
# usage: python benchmark_big.py
import numpy as np, os, subprocess, sys, time, csv

sys.path.insert(0, "C:/Users/qianl/OneDrive/codes/MARS/MARS_python")
from mars_py import mars_path, findA

RS = "C:/Users/qianl/R/R-4.6.1/bin/Rscript.exe"
HERE = "C:/Users/qianl/OneDrive/codes/MARS/MARS_python"
LAMS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
THR = 1e-4

def kkt(Om, X, lam):
    """KKT residual without forming S=AA' explicitly: h = 0.5*(A@(A.T@Om) + (A@(A.T@Om)).T) - I"""
    A = findA(X); p = Om.shape[0]
    G = A @ (A.T @ Om)
    h = 0.5 * (G + G.T) - np.eye(p)
    Z = np.clip(-h / lam, -1, 1); np.fill_diagonal(Z, 0)
    return float(np.linalg.norm(h + lam * Z))

def run_dense(p, n, seed, tag):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, n))
    xcsv = os.path.join(HERE, "_data_%s.csv" % tag)
    np.savetxt(xcsv, X, delimiter=",", fmt="%.8g")
    out = os.path.join(HERE, "_r_%s" % tag)
    rp = subprocess.run([RS, os.path.join(HERE, "run_r_benchmark.R"), xcsv, out, "dense"],
                        capture_output=True, text=True, cwd=HERE)
    if rp.returncode != 0:
        print("R failed:", rp.stderr[-800:]); return []
    rpath = {}
    with open(out + "_path.csv") as f:
        for row in csv.DictReader(f):
            rpath[float(row["lambda"])] = float(row["time"])
    lam_r = sorted(rpath.keys(), reverse=True)
    Om_r = [np.loadtxt(out + "_Om_%d.csv" % k, delimiter=",") for k in range(1, len(lam_r) + 1)]
    t0 = time.perf_counter()
    Op, Lp, tp = mars_path(X, np.array(LAMS), stoptol=1e-4, maxiter=10, maxlambdacheck=True)
    tpy = time.perf_counter() - t0
    rows = []
    for lam in lam_r:
        iR = lam_r.index(lam); iP = list(Lp).index(lam) if lam in Lp else None
        if iP is None: continue
        OmS, OmN = Om_r[iR], Op[iP]
        rel = np.linalg.norm(OmS - OmN) / max(1.0, np.linalg.norm(OmS))
        dmax = np.max(np.abs(OmS - OmN))
        support = np.mean((np.abs(OmS) > THR) == (np.abs(OmN) > THR))
        rS = kkt(OmS, X, lam); rN = kkt(OmN, X, lam)
        rows.append(dict(p=p, n=n, seed=seed, lam=lam, rel=rel, dmax=dmax, support=support,
                         kktR=rS, kktPy=rN, timeR=rpath[lam], timePy=tp[iP],
                         nnzR=int(np.sum(np.abs(OmS) > 1e-8)), nnzPy=int(np.sum(np.abs(OmN) > 1e-8))))
    os.remove(xcsv)
    for f in os.listdir(HERE):
        if f.startswith("_r_%s" % tag): os.remove(os.path.join(HERE, f))
    return rows

def run_sparse(p, n, seed, tag):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, n))
    xcsv = os.path.join(HERE, "_data_%s.csv" % tag)
    np.savetxt(xcsv, X, delimiter=",", fmt="%.8g")
    out = os.path.join(HERE, "_r_%s" % tag)
    rp = subprocess.run([RS, os.path.join(HERE, "run_r_benchmark.R"), xcsv, out, "sparse"],
                        capture_output=True, text=True, cwd=HERE)
    if rp.returncode != 0:
        print("R failed:", rp.stderr[-800:]); return []
    rpath = {}
    with open(out + "_path.csv") as f:
        for row in csv.DictReader(f):
            rpath[float(row["lambda"])] = float(row["time"])
    lam_r = sorted(rpath.keys(), reverse=True)
    R_sp = []
    for k in range(1, len(lam_r) + 1):
        tri = np.loadtxt(out + "_Om_%d_sp.csv" % k, delimiter=",")
        if tri.ndim == 1: tri = tri.reshape(1, -1)
        i = (tri[:, 0].astype(int) - 1); j = (tri[:, 1].astype(int) - 1); v = tri[:, 2]
        R_sp.append((i, j, v))
    t0 = time.perf_counter()
    Op, Lp, tp = mars_path(X, np.array(LAMS), stoptol=1e-4, maxiter=10, maxlambdacheck=True)
    tpy = time.perf_counter() - t0
    rows = []
    for lam in lam_r:
        iR = lam_r.index(lam); iP = list(Lp).index(lam) if lam in Lp else None
        if iP is None: continue
        i, j, v = R_sp[iR]
        OmN = Op[iP]
        Rsup = (i, j)  # set-like: use dict below
        Rd = {}
        for ii, jj, vv in zip(i, j, v):
            Rd[(int(ii), int(jj))] = vv
        # Py support (upper tri)
        absN = np.abs(OmN)
        ii2, jj2 = np.nonzero(np.triu(absN > 1e-8))
        keysR = set(Rd.keys()); keysP = set(zip(ii2.tolist(), jj2.tolist()))
        both = keysR & keysP
        onlyR = len(keysR - keysP); onlyP = len(keysP - keysR)
        sup_cov = len(both) / max(1, len(keysR))
        # value diff on common support
        d = [abs(Rd[k] - OmN[k]) for k in list(both)[:20000]]
        dmax = max(d) if d else 0.0
        vrel = float(np.sqrt(sum(x * x for x in d)) / max(1.0, np.sqrt(sum(Rd[k] * Rd[k] for k in both))))
        rS = float("nan")   # R KKT not exported for sparse mode
        rN = kkt(OmN, X, lam)
        rows.append(dict(p=p, n=n, seed=seed, lam=lam, rel=vrel, dmax=dmax,
                         support=sup_cov, kktR=rS, kktPy=rN,
                         timeR=rpath[lam], timePy=tp[iP],
                         nnzR=len(keysR), nnzPy=len(keysP)))
    os.remove(xcsv)
    for f in os.listdir(HERE):
        if f.startswith("_r_%s" % tag): os.remove(os.path.join(HERE, f))
    return rows

def main():
    cases = [(1000, 50, 6, "f", "dense"), (5000, 50, 7, "g", "sparse")]
    allrows = []
    for p, n, seed, tag, mode in cases:
        print("== case p=%d n=%d (%s) ==" % (p, n, mode), flush=True)
        rows = run_dense(p, n, seed, tag) if mode == "dense" else run_sparse(p, n, seed, tag)
        for r in rows:
            print("  lam=%.2f rel=%.2e dmax=%.2e support=%.4f tR=%.3f tPy=%.3f nnzR=%d nnzPy=%d" %
                  (r["lam"], r["rel"], r["dmax"], r["support"], r["timeR"], r["timePy"], r["nnzR"], r["nnzPy"]))
        allrows += rows
    with open(os.path.join(HERE, "benchmark_big_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)
    print("DONE")

if __name__ == "__main__":
    main()
