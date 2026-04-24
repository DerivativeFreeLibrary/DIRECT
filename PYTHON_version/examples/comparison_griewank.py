"""
Confronto delle performance tra direct_opt e scipy.optimize.direct
sulla funzione di Griewank traslata in 5 dimensioni.

La traslazione sposta il minimo globale da 0 a una posizione arbitraria
(shift = 123.45), verificando la capacita' dell'algoritmo di localizzare
ottimi non centrati nel dominio di ricerca.
"""

import time
import numpy as np
from scipy.optimize import direct as scipy_direct
from direct_opt import direct_optimize_batch

# Traslazione del minimo: il minimo globale si trova in x = [shift, ..., shift]
SHIFT = 123.45


def griewank_shifted(X):
    """
    Funzione di Griewank traslata vettorializzata.

    Minimo globale: f(SHIFT, ..., SHIFT) = 0.0
    Dominio: x_i in [-600, 600]

    Parametri:
    ----------
    X : ndarray di shape (n_points, n_dims)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    X_s = X - SHIFT
    sum_term = np.sum(X_s**2, axis=1) / 4000.0
    dim_idx = np.arange(1, X.shape[1] + 1)
    prod_term = np.prod(np.cos(X_s / np.sqrt(dim_idx)), axis=1)
    return sum_term - prod_term + 1.0


def griewank_shifted_scalar(x):
    """Wrapper scalare per scipy.optimize.direct (accetta un singolo punto 1D)."""
    return griewank_shifted(x[np.newaxis, :])[0]


if __name__ == "__main__":
    n_dims = 5
    bounds = [(-600.0, 600.0)] * n_dims
    f_opt  = 0.0
    BUDGET = 20000

    print(f"Funzione      : Griewank traslata ({n_dims}D), shift = {SHIFT}")
    print(f"Ottimo teorico: f = {f_opt},  x = [{SHIFT}, ...]")
    print(f"Budget        : {BUDGET} valutazioni\n")

    # --- direct_opt ---
    t0 = time.perf_counter()
    x_do, f_do, n_do = direct_optimize_batch(
        griewank_shifted,
        bounds,
        max_evals=BUDGET,
        max_iter=1000,
        eps=1e-4
    )
    t_do = time.perf_counter() - t0

    # --- scipy.optimize.direct ---
    t0 = time.perf_counter()
    res = scipy_direct(griewank_shifted_scalar, bounds, maxfun=BUDGET)
    t_scipy = time.perf_counter() - t0

    # --- Risultati ---
    header = f"{'Metodo':<20} | {'f_best':>15} | {'Valutazioni':>12} | {'Tempo (s)':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'direct_opt':<20} | {f_do:>15.8e} | {n_do:>12} | {t_do:>10.4f}")
    print(f"{'scipy.direct':<20} | {res.fun:>15.8e} | {res.nfev:>12} | {t_scipy:>10.4f}\n")
    
    # Sistemo il print in modo che sia inequivocabile
    speedup = t_scipy / t_do
    print(f"Risultato: direct_opt è {speedup:.2f} volte più veloce di scipy.direct")