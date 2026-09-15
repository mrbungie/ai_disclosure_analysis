"""Archetypal Analysis fitted from given initial archetypes (temporal warm start).

X ~ A Z with Z = B X; rows of A (n x k) and B (k x n) on the probability
simplex, minimizing (1/n)||X - A Z||^2 by alternating projected gradient
steps. `fit(X, Z_init)` starts from the archetypes of the previous snapshot
expressed in the current standardized space, so consecutive snapshots follow
the same local optimum instead of jumping between near-equivalent solutions.
"""
from __future__ import annotations

import numpy as np


def project_simplex_rows(V: np.ndarray) -> np.ndarray:
    n, m = V.shape
    U = np.sort(V, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1
    idx = np.arange(1, m + 1)
    cond = U - css / idx > 0
    rho = cond.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0)


def simplex_lstsq(X: np.ndarray, Z: np.ndarray, iters: int = 500) -> np.ndarray:
    """argmin_A ||X - A Z||^2 with each row of A on the simplex."""
    A = np.full((X.shape[0], Z.shape[0]), 1.0 / Z.shape[0])
    step = 1.0 / (2 * np.linalg.norm(Z @ Z.T, 2) + 1e-12)
    for _ in range(iters):
        A = project_simplex_rows(A + step * 2 * (X - A @ Z) @ Z.T)
    return A


class WarmStartAA:
    def __init__(self, max_iter: int = 2000, tol: float = 1e-7):
        self.max_iter, self.tol = max_iter, tol

    @staticmethod
    def _fit_loss(X, A, B):
        return float(np.sum((X - A @ (B @ X)) ** 2) / X.shape[0])

    def fit(self, X: np.ndarray, Z_init: np.ndarray) -> "WarmStartAA":
        n = X.shape[0]
        B = simplex_lstsq(Z_init, X, iters=800)
        A = simplex_lstsq(X, B @ X, iters=300)
        prev = self._fit_loss(X, A, B)
        step_B = 1.0
        for it in range(self.max_iter):
            Z = B @ X
            step_A = 1.0 / (2.0 / n * np.linalg.norm(Z @ Z.T, 2) + 1e-12)
            for _ in range(5):
                A = project_simplex_rows(A + step_A * 2.0 / n * (X - A @ Z) @ Z.T)
            G = (-2.0 / n * A.T @ (X - A @ Z)) @ X.T
            cur = self._fit_loss(X, A, B)
            while True:
                B_new = project_simplex_rows(B - step_B * G)
                new = self._fit_loss(X, A, B_new)
                if new <= cur - 1e-4 * np.sum(G * (B - B_new)) or step_B < 1e-10:
                    break
                step_B *= 0.5
            B, step_B = B_new, step_B * 1.5
            if abs(prev - new) < self.tol * max(1.0, abs(prev)):
                break
            prev = new
        self.B_, self.archetypes_ = B, B @ X
        self.fit_loss_, self.n_iter_ = self._fit_loss(X, A, B), it + 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return simplex_lstsq(X, self.archetypes_, iters=500)
