# Python/R benchmark: pure-python MARS vs R package MARS (SSNAL)
# steps: generate X -> run R (Rscript) -> run python -> compare solutions & timing
import numpy as np, os, subprocess, sys, time, csv

sys.path.insert(0, "C:/Users/qianl/OneDrive/codes/MARS/MARS_python")
from mars_py import mars_path, findA

RS = "C:/Users/qianl/R/R-4.6.1/bin/Rscript.exe"
HERE = "C:/Users/qianl/OneDrive/codes/MARS/MARS_python"
LAMS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def kkt(Om, X, lam):
    A = findA(X); S = A @ A.T; p = Om.shape[0]
    h = 0.5 * (S @ Om + Om @ S) - np.eye(p)
    Z = np.clip(-h / lam, -1, 1); np.fill_diagonal(Z, 0)
    return float(np.linalg.norm(h + lam * Z))

def run_case(p, n, seed, tag):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, n))
    xcsv = os.path.join(HERE, "_data_%s.csv" % tag)
    np.savetxt(xcsv, X, delimiter=",", fmt="%.10g")
    out = os.path.join(HERE, "_r_%s" % tag)

    # R side
    rp = subprocess.run([RS, os.path.join(HERE, "run_r_benchmark.R"), xcsv, out],
                        capture_output=True, text=True, cwd=HERE)
    if rp.returncode != 0:
        print("R failed:", rp.stderr[-500:]); return None
    rpath = {}
    with open(out + "_path.csv") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rpath[float(row["lambda"])] = float(row["time"])
    lam_r = sorted(rpath.keys(), reverse=True)
    Om_r = []
    for k in range(1, len(lam_r) + 1):
        Om_r.append(np.loadtxt(out + "_Om_%d.csv" % k, delimiter=","))

    # Python side
    t0 = time.perf_counter()
    Op, Lp, tp = mars_path(X, np.array(LAMS), stoptol=1e-4, maxiter=10,
                           printyes=False, maxlambdacheck=True)
    tpy = time.perf_counter() - t0
    # align by lambda
    rows = []
    for lam in lam_r:
        iR = lam_r.index(lam)
        iP = list(Lp).index(lam) if lam in Lp else None
        if iP is None: continue
        OmS = Om_r[iR]; OmN = Op[iP]
        rel = np.linalg.norm(OmS - OmN) / max(1.0, np.linalg.norm(OmS))
        dmax = np.max(np.abs(OmS - OmN))
        thr = 1e-4
        support = np.mean((np.abs(OmS) > thr) == (np.abs(OmN) > thr))
        rS = kkt(OmS, X, lam); rN = kkt(OmN, X, lam)
        rows.append(dict(p=p, n=n, seed=seed, lam=lam, rel=rel, dmax=dmax,
                         support=support, kktR=rS, kktPy=rN,
                         timeR=rpath[lam], timePy=tp[iP]))
    os.remove(xcsv)
    for f in os.listdir(HERE):
        if f.startswith("_r_%s" % tag):
            os.remove(os.path.join(HERE, f))
    return rows

def main():
    cases = [(30, 10, 1, "a"), (100, 30, 2, "b"), (200, 50, 3, "c"),
             (500, 80, 4, "d"), (100, 50, 5, "e")]
    allrows = []
    for p, n, seed, tag in cases:
        print("== case p=%d n=%d ==" % (p, n), flush=True)
        rows = run_case(p, n, seed, tag)
        if rows is None: continue
        for r in rows:
            print("  lam=%.2f rel=%.2e support=%.4f kkt[R]=%.1e kkt[Py]=%.1e tR=%.3f tPy=%.3f"
                  % (r["lam"], r["rel"], r["support"], r["kktR"], r["kktPy"], r["timeR"], r["timePy"]))
        allrows += rows
    with open(os.path.join(HERE, "benchmark_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)
    print("\nsummary (per case): total time R vs Py, mean rel")
    for p, n, seed, tag in cases:
        rs = [r for r in allrows if r["p"] == p and r["n"] == n]
        if rs:
            print("p=%d n=%d: tR=%.3fs tPy=%.3fs mean_rel=%.2e" %
                  (p, n, sum(r["timeR"] for r in rs), sum(r["timePy"] for r in rs),
                   np.mean([r["rel"] for r in rs])))
    print("DONE")

if __name__ == "__main__":
    main()
