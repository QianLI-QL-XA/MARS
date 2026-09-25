# Pure Python (numpy) implementation of MARS
# Faithful port of src/MARSc.cpp (SSNAL: adaptive sieving + semismooth Newton/ALM + PCG),
# problem (6):  min_{Om in S^p} 0.5||Om A||_F^2 - <Om,I> + lam||Om||_{1,off}
# Usage: Omegapath, Lambdapath, timepath = mars_path(X, Lambdapath, ...)
import numpy as np
import time as _time

EPS = 1e-15

# ===================== numba JIT kernels =====================
USE_NUMBA = True
try:
    from numba import njit

    @njit(cache=True)
    def _sumsq(X):
        Xf = X.reshape(-1)
        s = 0.0
        for idx in range(Xf.shape[0]):
            s += Xf[idx] * Xf[idx]
        return s

    @njit(cache=True)
    def _normF(X):
        return np.sqrt(_sumsq(X))

    @njit(cache=True)
    def _opSY_jit(Y, A, i, j):
        """v_k = sum(Y_i .* A_j + A_i .* Y_j)"""
        t = i.shape[0]
        n = Y.shape[1]
        out = np.zeros(t)
        for k in range(t):
            ik, jk = i[k], j[k]
            s = 0.0
            for l in range(n):
                s += Y[ik, l] * A[jk, l] + A[ik, l] * Y[jk, l]
            out[k] = s
        return out

    @njit(cache=True)
    def _opInvLA_jit(x, A, i, j):
        """OA = L_A^{-1}(x): diag x_k*A_i; offdiag .5 x_k*A_j / .5 x_k*A_i"""
        p, n = A.shape
        OA = np.zeros((p, n))
        for k in range(i.shape[0]):
            ik, jk = i[k], j[k]
            xk = x[k]
            if ik == jk:
                for l in range(n):
                    OA[ik, l] += xk * A[ik, l]
            else:
                for l in range(n):
                    OA[ik, l] += 0.5 * xk * A[jk, l]
                    OA[jk, l] += 0.5 * xk * A[ik, l]
        return OA

    @njit(cache=True)
    def _prox_b_jit(z, lam, d):
        v = np.zeros_like(z)
        for k in range(z.shape[0]):
            if abs(d[k]) > EPS:
                v[k] = min(max(z[k], -lam), lam)
        return v

    @njit(cache=True)
    def _partgradient_jit(zin, c, lam):
        u = np.ones(zin.shape[0])
        for k in range(u.shape[0]):
            if abs(c[k]) <= EPS and abs(zin[k]) <= lam:
                u[k] = 0.0
        return u

    @njit(cache=True)
    def _PMEASCG_jit(res, tolCG, maxiterCG, A, i, j, u, a, sigma):
        """plain CG on (I + sigma L_A^{-1} diag(u) L_A), as in C++ PMEASCG"""
        g = res.copy()
        err0 = _normF(res)
        rz1 = err0 * err0
        rz2 = 1.0
        direction = np.zeros_like(res)
        solveok = 1
        win = np.zeros(11)
        win[0] = err0
        for it in range(maxiterCG):
            if it > 0:
                beta = rz1 / rz2
                g = res + beta * g
            ldua = 0.5 * _opSY_jit(g, A, i, j)
            for k in range(ldua.shape[0]):
                ldua[k] *= u[k]
            Vg = g + sigma * _opInvLA_jit(ldua, A, i, j)
            denom = _sumsq_xy(g, Vg)
            if abs(denom) < EPS:
                solveok = 2
                break
            alpha = rz1 / denom
            direction = direction + alpha * g
            res = res - alpha * Vg
            residual = _normF(res)
            win[(it + 1) % 11] = residual
            if residual < tolCG:
                break
            rz2 = rz1
            rz1 = _sumsq(res)
            if it > 20:
                mn = 1e300
                mx = -1e300
                for ii in range(10):
                    r = win[(it - ii + 1) % 11] / win[(it - ii) % 11]
                    if r < mn:
                        mn = r
                    if r > mx:
                        mx = r
                if mn > 0.997 and mx < 1.003:
                    solveok = -1
                    break
        return direction, solveok

    @njit(cache=True)
    def _findstep_jit(GradPsiY, steptol, stepop, sigma, direction, A, i, j, Y, ztmp, z, PsiY, a, d, lam):
        """Armijo/line search on Psi (as in C++ findstep)"""
        maxiterstep = 17  # ceil(log2(1/steptol))
        c1 = 1e-4
        c2 = 0.9
        change0 = -_sumsq_xy(GradPsiY, direction)
        if change0 >= 0:
            return Y, ztmp, z, PsiY, 1.0
        alpconst = 0.5
        LB = 0.0
        UB = 1.0
        PsiYold = PsiY
        Yold = Y.copy()
        zold = z.copy()
        ztmpold = ztmp.copy()
        gLB = change0
        gUB = change0
        alp = 1.0
        for it in range(maxiterstep):
            if it > 0:
                alp = alpconst * (LB + UB)
            Y = Yold + alp * direction
            zin = ztmpold + zold - 0.5 * alp * _opSY_jit(direction, A, i, j)
            z = _prox_b_jit(zin, lam, d)
            ztmp = zin - z
            PsiY = 0.5 * _sumsq(Y) + 0.5 * sigma * _sumsq(ztmp)
            gradstep1 = Y - sigma * _opInvLA_jit(ztmp, A, i, j)
            change1 = alp * _sumsq_xy(gradstep1, direction)
            if it == 0:
                gUB = change1
                if (gLB > 0 and gUB > 0) or (gLB < 0 and gUB < 0):
                    break
            if abs(change1) < c2 * abs(change0) and (PsiY - PsiYold - c1 * alp * change0) <= 1e-8 / max(1.0, abs(PsiYold)):
                if stepop == 1 or (stepop == 2 and abs(change1) < steptol):
                    break
            if (change1 > 0 and gUB < 0) or (change1 < 0 and gUB > 0):
                LB = alp
                gLB = change1
            elif (change1 > 0 and gLB < 0) or (change1 < 0 and gLB > 0):
                UB = alp
                gUB = change1
        return Y, ztmp, z, PsiY, alp

    @njit(cache=True)
    def _sumsq_xy(X, Ymat):
        s = 0.0
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                s += X[r, c] * Ymat[r, c]
        return s

    @njit(cache=True)
    def _proxBmain_jit(XX, lam):
        """symmetric clamp of off-diagonal entries to [-lam, lam]; diagonal untouched."""
        p = XX.shape[0]
        P = np.zeros_like(XX)
        for i in range(p):
            for j in range(i + 1, p):
                v = XX[i, j]
                if v > lam:
                    v = lam
                elif v < -lam:
                    v = -lam
                P[i, j] = v
                P[j, i] = v
        return P

    @njit(cache=True)
    def _nonzero_upper_jit(Om, thresh):
        """column-major (i,j) pairs of upper triangle (incl. diag) with |Om| > thresh, plus values."""
        p = Om.shape[0]
        n = 0
        for i in range(p):
            for j in range(i, p):
                if abs(Om[i, j]) > thresh:
                    n += 1
        ii = np.zeros(n, np.int64)
        jj = np.zeros(n, np.int64)
        vv = np.zeros(n)
        k = 0
        for i in range(p):
            for j in range(i, p):
                if abs(Om[i, j]) > thresh:
                    ii[k] = i; jj[k] = j; vv[k] = Om[i, j]; k += 1
        return ii, jj, vv

    @njit(cache=True)
    def _gradP_jit(Om, G, ii, jj, vv):
        """0.5*(Om@G + G@Om) with Om sparse-symmetric (G = A@A.T precomputed).
        Math: (Om@A)@A.T + A@(A.T@Om) = Om@G + G@Om by associativity."""
        p = Om.shape[0]
        H = np.zeros((p, p))
        for k in range(ii.shape[0]):
            i = ii[k]; j = jj[k]; v = vv[k]
            if i == j:
                for c in range(p):
                    H[i, c] += v * G[i, c]
                    H[c, i] += v * G[c, i]
            else:
                for c in range(p):
                    H[i, c] += v * G[j, c]
                    H[j, c] += v * G[i, c]
                    H[c, j] += v * G[c, i]
                    H[c, i] += v * G[c, j]
        return 0.5 * H

    @njit(cache=True)
    def _tri_flat_gt_jit(M, thresh):
        """column-major flat indices (i + j*p) of strictly-upper triangle |M| > thresh."""
        p = M.shape[0]
        n = 0
        for i in range(p):
            for j in range(i + 1, p):
                if abs(M[i, j]) > thresh:
                    n += 1
        out = np.zeros(n, np.int64)
        k = 0
        for i in range(p):
            for j in range(i + 1, p):
                if abs(M[i, j]) > thresh:
                    out[k] = i + j * p
                    k += 1
        return out

    @njit(cache=True)
    def _maxlam_fromG_jit(G, Sdiag):
        """max over i<j of 0.5*|G_ij|*(Sii+Sjj)/(Sii*Sjj) using the full Gram G (findmaxlambda)."""
        p = G.shape[0]
        m = 0.0
        for i in range(p):
            si = Sdiag[i]
            for j in range(i + 1, p):
                val = 0.5 * abs(G[i, j]) * (si + Sdiag[j]) / (si * Sdiag[j])
                if val > m:
                    m = val
        return m

    @njit(cache=True)
    def _norm_sym_jit(M):
        """Frobenius norm of a symmetric matrix, exploiting symmetry (upper-tri angle only)."""
        p = M.shape[0]
        s = 0.0
        for i in range(p):
            s += M[i, i] * M[i, i]
        for i in range(p):
            for j in range(i + 1, p):
                s += 2.0 * M[i, j] * M[i, j]
        return np.sqrt(s)

except ImportError:
    USE_NUMBA = False
# ===================== end numba kernels =====================


def findA(X):
    """replicate src/MARSc.cpp findA(): row-center + thin SVD, A = U diag(s)/sqrt(n-1)"""
    p, n = X.shape
    Z = X - X.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    A = np.zeros((p, n))
    k = min(p, n)
    A[:, :k] = U[:, :k] * s[:k] / np.sqrt(n - 1)
    return A


def findmaxlambda(A, blocksize=400):
    """max over off-diagonal 0.5*|S_ij/S_ii + S_ij/S_jj|, also returns diag(S).
    Blocked computation: never materializes the full p x p Gram matrix (C++ style)."""
    p = A.shape[0]
    Sdiag = np.einsum('ij,ij->i', A, A)
    maxlam = 0.0
    for j0 in range(1, p, blocksize):
        j1 = min(j0 + blocksize, p)
        Scol = A @ A[j0:j1, :].T           # p x (j1-j0) Gram block
        sdc = Sdiag[j0:j1]
        ii = np.arange(j1)[:, None]
        jj = (np.arange(j1 - j0)[None, :] + j0)
        mask = ii < jj                      # strictly upper triangle i < j
        num = Scol[:j1][mask]
        si = np.broadcast_to(Sdiag[:j1, None], (j1, j1 - j0))[mask]
        sj = np.broadcast_to(sdc[None, :], (j1, j1 - j0))[mask]
        val = 0.5 * np.abs(num) * (si + sj) / (si * sj)        # 0.5*|S_ij/S_ii + S_ij/S_jj|
        if val.size:
            m = float(np.max(val))
            if m > maxlam: maxlam = m
    return maxlam, Sdiag


def proxBmain(XX, lam):
    """clamp off-diagonal entries of XX to [-lam, lam], diagonal untouched (returns 0 diag)"""
    if USE_NUMBA:
        return _proxBmain_jit(XX, lam)
    p = XX.shape[0]
    P = np.zeros_like(XX)
    if p > 1:
        cl = np.clip(np.triu(XX, 1), -lam, lam)
        P += cl + cl.T
    return P


def ind2sub(idx, shape):
    """arma column-major linear index -> (rows, cols)"""
    p, _ = shape
    idx = np.asarray(idx, dtype=np.int64)
    rows = idx % p
    cols = idx // p
    return rows, cols


def findcd(sub):
    """c_k=1, d_k=0 for diagonal; c_k=0, d_k=2 otherwise"""
    t = sub.shape[1]
    c = np.zeros(t); d = np.full(t, 2.0)
    m = sub[0] == sub[1]
    c[m] = 1.0; d[m] = 0.0
    return c, d


def vecOmega(Om, sub):
    return Om[sub[0], sub[1]]


def operatorSY(Y, A, sub):
    """v_k = sum(Y_i .* A_j + A_i .* Y_j) for active (i,j)"""
    if USE_NUMBA:
        return _opSY_jit(Y, A, sub[0], sub[1])
    i = sub[0]; j = sub[1]
    return np.sum(Y[i] * A[j] + A[i] * Y[j], axis=1)


def operatorInvLA(x, A, sub):
    """OA = L_A^{-1}(x): OA_i += x_k A_i (diag); OA_i += .5 x_k A_j, OA_j += .5 x_k A_i (offdiag)"""
    if USE_NUMBA:
        return _opInvLA_jit(x, A, sub[0], sub[1])
    p, n = A.shape
    i = sub[0]; j = sub[1]
    OA = np.zeros((p, n))
    diagm = i == j
    if diagm.any():
        np.add.at(OA, i[diagm], x[diagm, None] * A[i[diagm]])
    offm = ~diagm
    if offm.any():
        np.add.at(OA, i[offm], 0.5 * x[offm, None] * A[j[offm]])
        np.add.at(OA, j[offm], 0.5 * x[offm, None] * A[i[offm]])
    return OA


def prox_b(z, lam, d):
    if USE_NUMBA:
        return _prox_b_jit(z, lam, d)
    v = np.zeros_like(z)
    m = np.abs(d) > EPS
    v[m] = np.clip(z[m], -lam, lam)
    return v


def partgradient(zin, c, lam):
    if USE_NUMBA:
        return _partgradient_jit(zin, c, lam)
    u = np.ones(len(zin))
    m = (np.abs(c) <= EPS) & (np.abs(zin) <= lam)
    u[m] = 0.0
    return u


def updatesigma(primwin, dualwin, sigma, itersub, subbreakyes):
    minsig = 1e-4; maxsig = 1e7
    if itersub < 10: sigiter = 2
    elif itersub < 200: sigiter = 3
    elif itersub < 500: sigiter = 10
    else: sigiter = 20
    if (itersub % sigiter == 0) and subbreakyes < 0:
        mult = 5.0
        if primwin > max(1.0, 1.2 * dualwin):
            primwin = 0; sigma = min(maxsig, mult * sigma)
        elif dualwin > max(1.0, 1.2 * primwin):
            dualwin = 0; sigma = max(minsig, sigma / mult)
    if subbreakyes >= 0 and itersub > 10:
        sigma = max(minsig, sigma / 2.5)
    return primwin, dualwin, sigma


def PMEASCG(res, tolCG, maxiterCG, A, sub, u, a, sigma):
    """preconditioned? plain CG on (I + sigma L_A^{-1} diag(u) L_A) system (as in C++)"""
    if USE_NUMBA:
        direction, solveok = _PMEASCG_jit(res, tolCG, maxiterCG, A, sub[0], sub[1], u, a, sigma)
        return direction, solveok, []
    err = [float(np.linalg.norm(res))]
    g = res.copy()
    rz1 = err[0] ** 2; rz2 = 1.0
    direction = np.zeros_like(res)
    solveok = 1
    for it in range(maxiterCG):
        if it > 0:
            beta = rz1 / rz2
            g = res + beta * g
        ldua = 0.5 * operatorSY(g, A, sub)
        ldua = ldua * u
        Vg = g + sigma * operatorInvLA(ldua, A, sub)
        denom = float(np.sum(g * Vg))
        if abs(denom) < EPS:
            solveok = 2; break
        alpha = rz1 / denom
        direction = direction + alpha * g
        res = res - alpha * Vg
        residual = float(np.linalg.norm(res))
        err.append(residual)
        if residual < tolCG: break
        rz2 = rz1
        rz1 = float(np.sum(res * res))
        if it > 20:
            ratio = np.array([err[it - i + 1] / err[it - i] for i in range(10)])
            if ratio.min() > 0.997 and ratio.max() < 1.003:
                solveok = -1; break
    return direction, solveok, err


def findstep(GradPsiY, steptol, stepop, sigma, direction, A, Y, ztmp, z, PsiY, a, d, sub, lam):
    if USE_NUMBA:
        return _findstep_jit(GradPsiY, steptol, stepop, sigma, direction, A, sub[0], sub[1],
                             Y, ztmp, z, PsiY, a, d, lam)
    maxiterstep = int(np.ceil(np.log2(1.0 / (steptol + EPS))))
    c1 = 1e-4; c2 = 0.9
    change0 = float(np.sum(-GradPsiY * direction))
    if change0 >= 0:
        return Y, ztmp, z, PsiY, 1.0
    alpconst = 0.5; LB = 0.0; UB = 1.0
    PsiYold = PsiY; Yold = Y.copy(); zold = z.copy(); ztmpold = ztmp.copy()
    gLB = change0; gUB = change0; alp = 1.0
    for it in range(maxiterstep):
        if it > 0:
            alp = alpconst * (LB + UB)
        Y = Yold + alp * direction
        zin = ztmpold + zold - 0.5 * alp * operatorSY(direction, A, sub)
        z = prox_b(zin, lam, d)
        ztmp = zin - z
        PsiY = 0.5 * float(np.sum(Y * Y)) + 0.5 * sigma * float(np.sum(ztmp * ztmp))
        gradstep1 = Y - sigma * operatorInvLA(ztmp, A, sub)
        change1 = float(np.sum(gradstep1 * (alp * direction)))
        if it == 0:
            gUB = change1
            if np.sign(gLB) * np.sign(gUB) > 0:
                break
        if abs(change1) < c2 * abs(change0) and (PsiY - PsiYold - c1 * alp * change0) <= 1e-8 / max(1.0, abs(PsiYold)):
            if stepop == 1 or (stepop == 2 and abs(change1) < steptol):
                break
        if np.sign(change1) * np.sign(gUB) < 0:
            LB = alp; gLB = change1
        elif np.sign(change1) * np.sign(gLB) < 0:
            UB = alp; gUB = change1
    return Y, ztmp, z, PsiY, alp


def PMEASSSNCGc(Y, A, x, lam, sigma, maxitersub, Stolconst, stoptol, p, n, Index, sub, a, b, c, d):
    """semismooth Newton subproblem (C++ PMEASSSNCGc); returns z, ztmp, SY, subbreakyes"""
    maxiterCG = 500
    SY = 0.5 * operatorSY(Y, A, sub)
    zin = x / sigma - SY + c
    z = prox_b(zin, lam, d)
    ztmp = zin - z
    PsiY = 0.5 * float(np.sum(Y * Y)) + 0.5 * sigma * float(np.sum(ztmp * ztmp))
    subprimfeas_l = []
    subdualfeas_l = []
    subbreakyes = 0
    for itersub in range(maxitersub):
        GradPsiY = sigma * operatorInvLA(ztmp, A, sub) - Y
        subprimfeas = float(np.linalg.norm(GradPsiY)) / (1 + float(np.linalg.norm(Y)))
        subdualfeas = float(np.linalg.norm(SY + z - c)) / (1 + np.sqrt(p))
        tolsubconst = 0.9 if max(subprimfeas, subdualfeas) < stoptol else 0.05
        stoptolsub = max(min(1.0, Stolconst * subdualfeas), stoptol * tolsubconst)
        subprimfeas_l.append(subprimfeas); subdualfeas_l.append(subdualfeas)
        if subprimfeas < stoptolsub and itersub > 0:
            subbreakyes = -1
            break
        if subdualfeas > 1e-3 or itersub <= 5:
            maxiterCG = min(maxiterCG, 200)
        elif subdualfeas > 1e-4:
            maxiterCG = min(maxiterCG, 300)
        elif subdualfeas > 1e-5:
            maxiterCG = min(maxiterCG, 400)
        elif subdualfeas > 5e-6:
            maxiterCG = min(maxiterCG, 500)
        primratio = dualratio = 0.0
        if itersub > 0:
            primratio = subprimfeas / subprimfeas_l[itersub - 1]
            dualratio = subdualfeas / subdualfeas_l[itersub - 1]
        res = GradPsiY.copy()
        tolCG = min(5e-3, 0.1 * float(np.linalg.norm(res)))
        tolCGconst = 1.0
        if itersub > 0 and (subprimfeas > 0.1 * subprimfeas_l[0] or primratio > 0.5):
            tolCGconst = 0.5
        if dualratio > 1.1:
            tolCGconst = 0.5
        tolCG = tolCGconst * tolCG
        direction = np.zeros((p, n))
        u = partgradient(ztmp + z, c, lam)
        direction, solveok, _ = PMEASCG(res, tolCG, maxiterCG, A, sub, u, a, sigma)
        steptol = 1e-5
        stepop = 1 if (itersub < 2 or (itersub <= 2 and subdualfeas > 1e-4)) else 2
        alp = 1.0
        Y, ztmp, z, PsiY, alp = findstep(GradPsiY, steptol, stepop, sigma, direction, A, Y,
                                         ztmp, z, PsiY, a, d, sub, lam)
        SY = 0.5 * operatorSY(Y, A, sub)
        if alp < EPS:
            subbreakyes = 11
    return Y, z, ztmp, SY, subbreakyes


def PMEASmainc(A, lam, stoptol, maxiter, Index, sub, Omega, Y, p, n, sigma, printyessub=False):
    """ALM + semismooth Newton on active set (C++ PMEASmainc)"""
    lengthIndex = len(Index)
    maxitersub = 10
    Stolconst = 0.5
    primwin = 0; dualwin = 0
    itersub = 0
    breakyes = False
    c, d = findcd(sub)
    b = d + c
    a = 0.25 * d + c
    vomega = vecOmega(Omega, sub)
    x = vomega * b
    SY = 0.5 * operatorSY(Y, A, sub)
    SYc = SY - c
    OA = operatorInvLA(x, A, sub)
    primfeas = float(np.linalg.norm(Y - OA)) / (1 + float(np.linalg.norm(Y)))
    dualfeas = float(np.linalg.norm(SYc)) / (1 + np.sqrt(p))
    primobj = 0.5 * float(np.linalg.norm(OA) ** 2) - float(np.sum(vomega * c)) + lam * float(np.sum(np.abs(vomega * d)))
    dualobj = -0.5 * float(np.linalg.norm(Y) ** 2)
    gap = (primobj - dualobj) / (1 + abs(primobj) + abs(dualobj))
    eta = float(np.linalg.norm(SYc + prox_b(x - SYc, lam, d))) / (1 + float(np.linalg.norm(SYc)) + float(np.linalg.norm(x)))
    if max(primfeas, dualfeas) < 500 * max(1e-6, stoptol) and eta < stoptol:
        breakyes = True
    while itersub < maxiter and not breakyes:
        itersub += 1
        if dualfeas < 1e-3:
            maxitersub = max(maxitersub, 30)
        elif dualfeas < 1e-1:
            maxitersub = max(maxitersub, 20)
        Y, z, ztmp, SY, subbreakyes = PMEASSSNCGc(Y, A, x, lam, sigma, maxitersub, Stolconst,
                                                  stoptol, p, n, Index, sub, a, b, c, d)
        SYc = SY - c
        x = sigma * ztmp
        vomega = x * a
        OA = operatorInvLA(x, A, sub)
        primfeas = float(np.linalg.norm(Y - OA)) / (1 + float(np.linalg.norm(Y)))
        dualfeas = float(np.linalg.norm(SYc + z)) / (1 + np.sqrt(p))
        primobj = 0.5 * float(np.linalg.norm(OA) ** 2) - float(np.sum(vomega * c)) + lam * float(np.sum(np.abs(vomega * d)))
        dualobj = -0.5 * float(np.linalg.norm(Y) ** 2)
        gap = (primobj - dualobj) / (1 + abs(primobj) + abs(dualobj))
        eta = float(np.linalg.norm(SYc + prox_b(x - SYc, lam, d))) / (1 + float(np.linalg.norm(SYc)) + float(np.linalg.norm(x)))
        if max(primfeas, dualfeas) < 500 * max(1e-6, stoptol) and eta < stoptol:
            breakyes = True
        if primfeas < dualfeas:
            primwin += 1
        else:
            dualwin += 1
        primwin, dualwin, sigma = updatesigma(primwin, dualwin, sigma, itersub, subbreakyes)
    Om = np.zeros((p, p))
    nnzOmega = 0
    for k in range(lengthIndex):
        orow = int(sub[0, k]); ocol = int(sub[1, k])
        elemo = vomega[k]
        if abs(elemo) > EPS:
            nnzOmega += 1
            Om[orow, ocol] = elemo
            Om[ocol, orow] = elemo
    nnzOmega = 2 * (nnzOmega - p)
    return Om, Y, primobj, dualobj, gap, primfeas, dualfeas, eta, nnzOmega, sigma


def mars_path(X, Lambdapath, stoptol=1e-4, maxiter=10, stopmethod="fix",
              printyes=False, printyessub=False, sigma=1.0, maxlambdacheck=True):
    """main MARS path solver (C++ MARSc)"""
    stoptol = max(stoptol, 1e-6)
    p, n = X.shape
    A = findA(X)
    if USE_NUMBA:
        # full Gram G = A@A.T once; findmaxlambda scans it directly (associativity-equivalent)
        G = A @ A.T
        vecdiagS = G.diagonal().copy()
        maxlambda = float(_maxlam_fromG_jit(G, vecdiagS))
    else:
        G = None
        maxlambda, vecdiagS = findmaxlambda(A)
    Lambdapath = np.array(Lambdapath, dtype=float)
    if maxlambdacheck:
        Lambdapath = Lambdapath[Lambdapath < maxlambda]
        Lambdapath = np.sort(Lambdapath)[::-1]
    pathlength = len(Lambdapath)
    if pathlength == 0:
        return [], Lambdapath, []
    Omega = np.zeros((p, p))
    Y = A.copy()
    Index = np.arange(p, dtype=np.int64) * p + np.arange(p, dtype=np.int64)  # column-major diag
    for i in range(p):
        Omega[i, i] = 1.0 / vecdiagS[i]
        Y[i, :] /= vecdiagS[i]
    sub = ind2sub(Index, Omega.shape)
    rows = np.array(sub[0], dtype=np.int64); cols = np.array(sub[1], dtype=np.int64)
    sub = np.vstack([rows, cols])
    Omegapath = []
    timepath = []
    for iterpath in range(pathlength):
        lam = float(Lambdapath[iterpath])
        t0 = _time.perf_counter()
        if iterpath == 0:
            Omega, Y, primobj, dualobj, gap, primfeas, dualfeas, eta, nnzOmega, sigma = \
                PMEASmainc(A, lam, stoptol, maxiter, Index, sub, Omega, Y, p, n, sigma, printyessub)
        if USE_NUMBA:
            nz_i, nz_j, nz_v = _nonzero_upper_jit(Omega, 0.0)
            gradP = _gradP_jit(Omega, G, nz_i, nz_j, nz_v) - np.eye(p)
        else:
            gradP = 0.5 * ((Omega @ A) @ A.T + A @ (A.T @ Omega))
            gradP = gradP - np.eye(p)
        residual = gradP + proxBmain(Omega - gradP, lam)
        if USE_NUMBA:
            etaorg = float(_norm_sym_jit(residual)) / (1 + float(_norm_sym_jit(gradP)) + float(_norm_sym_jit(Omega)))
        else:
            etaorg = float(np.linalg.norm(residual)) / (1 + float(np.linalg.norm(gradP)) + float(np.linalg.norm(Omega)))
        numAS = 0
        while etaorg > stoptol and numAS < 10:
            numAS += 1
            nonzeronumberIndex = len(Index)
            # column-major flat indices of upper-triangle |residual| > 1e-5 (C++/Armadillo find())
            if USE_NUMBA:
                newJ = _tri_flat_gt_jit(residual, 1e-5)
            else:
                abstriures = np.abs(np.triu(residual))
                newJ_rowm = np.flatnonzero(abstriures > 1e-5)
                newJ = (newJ_rowm // p) + (newJ_rowm % p) * p
            newIndex = np.unique(np.concatenate([Index, newJ]))
            if len(newIndex) > 5 * nonzeronumberIndex and nonzeronumberIndex != 0:
                sortidx = np.argsort(-np.abs(residual), axis=None)   # flattened (row-major), descending
                sortidx = (sortidx // p) + (sortidx % p) * p   # row-major -> column-major
                nkeep = min(nonzeronumberIndex, len(sortidx) - 1)
                if nkeep > 0:
                    newJ = sortidx[:nkeep]
                    Index = np.unique(np.concatenate([Index, newJ]))
                else:
                    Index = newIndex
            else:
                Index = newIndex
            rows = Index % p; cols = Index // p
            sub = np.vstack([rows, cols])
            Omega, Y, primobj, dualobj, gap, primfeas, dualfeas, eta, nnzOmega, sigma = \
                PMEASmainc(A, lam, stoptol, maxiter, Index, sub, Omega, Y, p, n, sigma, printyessub)
            if USE_NUMBA:
                nz_i, nz_j, nz_v = _nonzero_upper_jit(Omega, 0.0)
                gradP = _gradP_jit(Omega, G, nz_i, nz_j, nz_v) - np.eye(p)
            else:
                gradP = 0.5 * ((Omega @ A) @ A.T + A @ (A.T @ Omega))
                gradP = gradP - np.eye(p)
            residual = gradP + proxBmain(Omega - gradP, lam)
            if USE_NUMBA:
                etaorg = float(_norm_sym_jit(residual)) / (1 + float(_norm_sym_jit(gradP)) + float(_norm_sym_jit(Omega)))
            else:
                etaorg = float(np.linalg.norm(residual)) / (1 + float(np.linalg.norm(gradP)) + float(np.linalg.norm(Omega)))
        timepath.append(_time.perf_counter() - t0)
        Omegapath.append(Omega.copy())
        if stopmethod == "bigs" and (nnzOmega > (len(Lambdapath) * n) or numAS > 4):
            Omegapath = Omegapath[:iterpath + 1]
            Lambdapath = Lambdapath[:iterpath + 1]
            timepath = timepath[:iterpath + 1]
            break
        if stopmethod == "fix" and iterpath == len(Lambdapath) - 1:
            break
    return Omegapath, Lambdapath, timepath


def maxLambda(X):
    """Data-dependent maximum lambda for a p x n sample matrix X.

    The largest tuning value for which a non-diagonal edge can enter the
    solution; candidate lambdas above it are dropped by ``maxlambdacheck``.
    """
    return findmaxlambda(findA(X))[0]
