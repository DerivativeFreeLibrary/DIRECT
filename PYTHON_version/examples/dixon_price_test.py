"""
Confronto delle performance tra direct_opt e scipy.optimize.direct
sulla funzione di Dixon-Price in 5 dimensioni.

La funzione di Dixon-Price presenta una valle stretta e curvilinea che
costituisce una sfida strutturale per l'algoritmo DIRECT, il cui meccanismo
di suddivisione rettangolare si adatta con difficolta' a geometrie non
allineate agli assi. Il confronto con scipy.optimize.direct consente di
verificare se le difficolta' osservate siano intrinseche al metodo o
specifiche di questa implementazione.
"""

import time
import numpy as np
from scipy.optimize import direct as scipy_direct
from direct_opt import direct_optimize_batch


def dixon_price(X):
    """
    Funzione di Dixon-Price vettorializzata.

    Minimo globale: f(x*) = 0.0
    con x*_i = 2^(-(2^i - 2) / 2^i)  per i = 1, ..., n
    Dominio standard: x_i in [-10, 10]

    Parametri:
    ----------
    X : ndarray di shape (n_points, n_dims)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    n_dims    = X.shape[1]
    term1     = (X[:, 0] - 1.0)**2
    indices   = np.arange(2, n_dims + 1)
    # sum_{i=2}^{n} i * (2*x_i^2 - x_{i-1})^2
    inner     = (2.0 * X[:, 1:]**2 - X[:, :-1])**2
    term_sum  = np.sum(indices * inner, axis=1)
    return term1 + term_sum


def dixon_price_scalar(x):
    """Wrapper scalare per scipy.optimize.direct (accetta un singolo punto 1D)."""
    return dixon_price(x[np.newaxis, :])[0]


if __name__ == "__main__":
    n_dims = 5
    bounds = [(-10.0, 10.0)] * n_dims
    f_opt  = 0.0
    BUDGET = 50000

    print(f"Funzione      : Dixon-Price ({n_dims}D)")
    print(f"Ottimo teorico: f = {f_opt}")
    print(f"Budget        : {BUDGET} valutazioni")
    print()

    # --- direct_opt ---
    t0 = time.perf_counter()
    x_do, f_do, n_do = direct_optimize_batch(
        dixon_price,
        bounds,
        max_evals=BUDGET,
        max_iter=2000,
        eps=1e-7
    )
    t_do = time.perf_counter() - t0

    # --- scipy.optimize.direct ---
    t0 = time.perf_counter()
    res = scipy_direct(dixon_price_scalar, bounds, maxfun=BUDGET)
    t_scipy = time.perf_counter() - t0

    # --- Risultati ---
    header = f"{'Metodo':<20} | {'f_best':>15} | {'Valutazioni':>12} | {'Tempo (s)':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'direct_opt':<20} | {f_do:>15.8e} | {n_do:>12} | {t_do:>10.4f}")
    print(f"{'scipy.direct':<20} | {res.fun:>15.8e} | {res.nfev:>12} | {t_scipy:>10.4f}")
    
    speedup = t_scipy / t_do
    print(f"Risultato: direct_opt è {speedup:.2f} volte più veloce di scipy.direct")
