"""
Confronto delle performance tra direct_opt e scipy.optimize.direct
sulla funzione di Levy 8 in 5 dimensioni.

Metrica di confronto: valore della funzione obiettivo, numero di valutazioni
e tempo di esecuzione, a parita' di budget massimo.
"""

import time
import numpy as np
from scipy.optimize import direct as scipy_direct
from direct_opt import direct_optimize_batch


def levy8(X):
    """
    Funzione di Levy (variante 8) vettorializzata.

    Minimo globale: f(1, ..., 1) = 0.0
    Dominio standard: x_i in [-10, 10]

    Parametri:
    ----------
    X : ndarray di shape (n_points, n_dims)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    n_dims = X.shape[1]
    f  = 10.0 * np.sin(np.pi * X[:, 0])**2
    f += np.sum((X[:, :-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * X[:, 1:])**2), axis=1)
    f += (X[:, -1] - 1.0)**2
    return f * (np.pi / n_dims)


def levy8_scalar(x):
    """Wrapper scalare per scipy.optimize.direct (accetta un singolo punto 1D)."""
    return levy8(x[np.newaxis, :])[0]


if __name__ == "__main__":
    n_dims  = 5
    bounds  = [(-10.0, 10.0)] * n_dims
    f_opt   = 0.0
    BUDGET  = 50000

    print(f"Funzione      : Levy 8 ({n_dims}D)")
    print(f"Ottimo teorico: f = {f_opt}")
    print(f"Budget        : {BUDGET} valutazioni")
    print()

    # --- direct_opt ---
    t0 = time.perf_counter()
    x_do, f_do, n_do = direct_optimize_batch(
        levy8,
        bounds,
        max_evals=BUDGET,
        max_iter=1000,
        eps=1e-4
    )
    t_do = time.perf_counter() - t0

    # --- scipy.optimize.direct ---
    t0 = time.perf_counter()
    res = scipy_direct(levy8_scalar, bounds, maxfun=BUDGET)
    t_scipy = time.perf_counter() - t0

    # --- Risultati ---
    header = f"{'Metodo':<20} | {'f_best':>15} | {'Valutazioni':>12} | {'Tempo (s)':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'direct_opt':<20} | {f_do:>15.8e} | {n_do:>12} | {t_do:>10.4f}")
    print(f"{'scipy.direct':<20} | {res.fun:>15.8e} | {res.nfev:>12} | {t_scipy:>10.4f}")
    
    speedup = t_scipy / t_do
    print(f"Risultato: direct_opt è {speedup:.2f} volte più veloce di scipy.direct")