"""
Confronto delle performance tra direct_opt e scipy.optimize.direct
sulla funzione di Schwefel in 5 dimensioni.

La funzione di Schwefel e' notoriamente difficile per gli algoritmi di
ottimizzazione globale perche' il minimo globale si trova in prossimita'
dei bordi del dominio (x_i ~ 420.97), lontano dal centro dove si
concentrano numerosi minimi locali profondi.
"""

import time
import numpy as np
from scipy.optimize import direct as scipy_direct
from direct_opt import direct_optimize_batch


def schwefel(X):
    """
    Funzione di Schwefel vettorializzata.

    Minimo globale: f(420.9687, ..., 420.9687) = 0.0
    Dominio standard: x_i in [-500, 500]

    Parametri:
    ----------
    X : ndarray di shape (n_points, n_dims)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    n_dims = X.shape[1]
    return 418.9829 * n_dims - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def schwefel_scalar(x):
    """Wrapper scalare per scipy.optimize.direct (accetta un singolo punto 1D)."""
    return schwefel(x[np.newaxis, :])[0]


if __name__ == "__main__":
    n_dims = 5
    bounds = [(-500.0, 500.0)] * n_dims
    f_opt  = 0.0
    BUDGET = 200000

    print(f"Funzione      : Schwefel ({n_dims}D)")
    print(f"Ottimo teorico: f = {f_opt},  x = [420.9687, ...]")
    print(f"Budget        : {BUDGET} valutazioni\n")

    # --- direct_opt ---
    t0 = time.perf_counter()
    x_do, f_do, n_do = direct_optimize_batch(
        schwefel,
        bounds,
        max_evals=BUDGET,
        max_iter=5000,
        eps=1e-6
    )
    t_do = time.perf_counter() - t0

    # --- scipy.optimize.direct ---
    t0 = time.perf_counter()
    res = scipy_direct(schwefel_scalar, bounds, maxfun=BUDGET)
    t_scipy = time.perf_counter() - t0

    # --- Risultati ---
    header = f"{'Metodo':<20} | {'f_best':>15} | {'Valutazioni':>12} | {'Tempo (s)':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'direct_opt':<20} | {f_do:>15.8e} | {n_do:>12} | {t_do:>10.4f}")
    print(f"{'scipy.direct':<20} | {res.fun:>15.8e} | {res.nfev:>12} | {t_scipy:>10.4f}\n")
    
    speedup = t_scipy / t_do
    print(f"Risultato: direct_opt è {speedup:.2f} volte più veloce di scipy.direct")
