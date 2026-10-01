"""Numerical checks for docs/hazard/design-note.md (NumPy only, no lightgbm-rust).

1. Worked examples: log score vs. ranked probability score on CIFs (sections 6.3 and 6.4).
2. Finite-difference checks of:
   - the per-row competing-risks gradient and 2x2 Hessian (section 4);
   - the CIF Jacobian dCIF_k(t)/da_j(u) (section 7.1);
   - the gradient of the uncensored RPS loss via suffix sums (section 7.2).

    uv run --no-sync python docs/hazard/check_formulas.py
"""

from __future__ import annotations

import numpy as np

K = 2  # event types: 0 = default (D), 1 = payoff (P); "continue" is the reference


def hazards(a: np.ndarray) -> np.ndarray:
    """a: (m, K) raw scores -> (m, K) cause-specific hazards (softmax with reference 0)."""
    e = np.exp(a - np.maximum(a.max(axis=1, keepdims=True), 0.0))
    ref = np.exp(-np.maximum(a.max(axis=1, keepdims=True), 0.0))
    return e / (ref + e.sum(axis=1, keepdims=True))


def survival_cif(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S[t] for t = 0..m (S[0] = 1) and CIF[t, k] for t = 1..m (index t-1)."""
    h0 = 1.0 - h.sum(axis=1)
    S = np.concatenate([[1.0], np.cumprod(h0)])
    cif = np.cumsum(S[:-1, None] * h, axis=0)
    return S, cif


def log_score(pmf: dict, horizon: int, outcome: tuple[int, int]) -> float:
    return float(-np.log(pmf[outcome]))


def rps_cif(pmf: dict, horizon: int, outcome: tuple[int, int]) -> float:
    """Sum over t = 1..horizon and k of (CIF_k(t) - 1[T <= t, type = k])^2 (no censoring)."""
    total = 0.0
    for k in range(K):
        cif = np.cumsum([pmf.get((t, k), 0.0) for t in range(1, horizon + 1)])
        y = np.array([1.0 if (outcome[1] == k and outcome[0] <= t) else 0.0 for t in range(1, horizon + 1)])
        total += float(((cif - y) ** 2).sum())
    return total


def examples() -> None:
    H = 36
    D, P = 0, 1
    # Forecast "spike": default concentrated in month 30, small background mass elsewhere.
    spike = {(t, D): 0.005 for t in range(1, H + 1)}
    spike.update({(t, P): 0.005 for t in range(1, H + 1)})
    spike[(30, D)] = 0.45
    # Forecast "smooth": the same total default mass near month 30 spread over months 29-31.
    smooth = dict(spike)
    smooth[(29, D)], smooth[(30, D)], smooth[(31, D)] = 0.155, 0.15, 0.155
    for name, f in (("spike", spike), ("smooth", smooth)):
        assert abs(sum(f.values())) < 1.0
        print(f"forecast {name}: P(default by {H}) = {sum(v for (t, k), v in f.items() if k == D):.3f}, "
              f"P(payoff by {H}) = {sum(v for (t, k), v in f.items() if k == P):.3f}")
        for label, outcome in (("default at 31", (31, D)), ("default at 5", (5, D)),
                               ("payoff at 30", (30, P)), ("default at 30", (30, D))):
            print(f"  truth {label:14s}: log score {log_score(f, H, outcome):6.3f}   "
                  f"RPS(CIF) {rps_cif(f, H, outcome):6.3f}")


def fd_checks(rng: np.random.Generator) -> None:
    eps = 1e-6
    # --- per-row likelihood gradient and Hessian
    for y in (0, 1, 2):  # 0 = continue, 1 = D, 2 = P
        a = rng.normal(size=K) * 2

        def nll(a_: np.ndarray) -> float:
            h = hazards(a_[None, :])[0]
            p = np.concatenate([[1 - h.sum()], h])
            return float(-np.log(p[y]))

        h = hazards(a[None, :])[0]
        g = h - np.array([y == 1, y == 2], dtype=float)
        Hm = np.diag(h) - np.outer(h, h)
        g_fd = np.array([(nll(a + eps * e) - nll(a - eps * e)) / (2 * eps) for e in np.eye(K)])
        e2 = 1e-4  # second differences need a larger step
        H_fd = np.array([[(nll(a + e2 * (ei + ej)) - nll(a + e2 * (ei - ej)) - nll(a - e2 * (ei - ej))
                           + nll(a - e2 * (ei + ej))) / (4 * e2**2) for ej in np.eye(K)] for ei in np.eye(K)])
        assert np.allclose(g, g_fd, atol=1e-7), (g, g_fd)
        assert np.allclose(Hm, H_fd, atol=1e-6), (Hm, H_fd)
    print("per-row likelihood gradient and 2x2 Hessian: finite differences agree")

    # --- CIF Jacobian
    m = 7
    a = rng.normal(size=(m, K)) - 2.0
    h = hazards(a)
    S, cif = survival_cif(h)
    for t in range(1, m + 1):
        for k in range(K):
            for u in range(1, m + 1):
                for j in range(K):
                    if u <= t:
                        cif_u1 = cif[u - 2, k] if u >= 2 else 0.0
                        an = (k == j) * S[u - 1] * h[u - 1, k] - h[u - 1, j] * (cif[t - 1, k] - cif_u1)
                    else:
                        an = 0.0
                    d = np.zeros_like(a)
                    d[u - 1, j] = eps
                    fd = (survival_cif(hazards(a + d))[1][t - 1, k] - survival_cif(hazards(a - d))[1][t - 1, k]) / (2 * eps)
                    assert abs(an - fd) < 1e-8, (t, k, u, j, an, fd)
    print("CIF Jacobian dCIF_k(t)/da_j(u): finite differences agree")

    # --- RPS gradient via suffix sums (uncensored loan observed over the whole horizon m)
    y = np.zeros((m, K))
    y[4:, 0] = 1.0  # default in month 5

    def rps(a_: np.ndarray) -> float:
        return float(((survival_cif(hazards(a_))[1] - y) ** 2).sum())

    r = 2 * (cif - y)                                    # r_k(t)
    R = np.cumsum(r[::-1], axis=0)[::-1]                 # R_k(u) = sum_{t>=u} r_k(t)
    Q = np.cumsum((r * cif)[::-1], axis=0)[::-1]         # Q_k(u) = sum_{t>=u} r_k(t) CIF_k(t)
    cif_prev = np.vstack([np.zeros(K), cif[:-1]])        # CIF_k(u-1)
    grad = np.zeros((m, K))
    for u in range(m):
        for j in range(K):
            grad[u, j] = S[u] * h[u, j] * R[u, j] - h[u, j] * sum(Q[u, k] - cif_prev[u, k] * R[u, k] for k in range(K))
    grad_fd = np.zeros_like(a)
    for u in range(m):
        for j in range(K):
            d = np.zeros_like(a)
            d[u, j] = eps
            grad_fd[u, j] = (rps(a + d) - rps(a - d)) / (2 * eps)
    assert np.allclose(grad, grad_fd, atol=1e-7), (grad, grad_fd)
    print("RPS gradient via suffix sums: finite differences agree")


if __name__ == "__main__":
    examples()
    fd_checks(np.random.default_rng(0))
