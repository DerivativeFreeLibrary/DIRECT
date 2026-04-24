import numpy as np
from direct_opt import direct_optimize_batch


def easom(X):
    """
    Funzione di Easom vettorializzata.

    Minimo globale: f(pi, pi) = -1.0
    Caratteristica: il minimo è localizzato in una regione molto ristretta
    rispetto all'ampiezza del dominio, il che la rende una funzione di test
    impegnativa per algoritmi di ottimizzazione globale.

    Parametri:
    ----------
    X : ndarray di shape (n_points, 2)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    x = X[:, 0]
    y = X[:, 1]
    cos_term = -np.cos(x) * np.cos(y)
    exp_term = np.exp(-((x - np.pi)**2 + (y - np.pi)**2))
    return cos_term * exp_term


if __name__ == "__main__":
    # Dominio esteso: il minimo in (pi, pi) occupa una frazione minima dello spazio
    bounds = [(-100.0, 100.0), (-100.0, 100.0)]

    x_best, f_best, n_rects = direct_optimize_batch(
        easom,
        bounds,
        max_iter=100,
        max_evals=10000
    )

    print(f"Minimo trovato    : x = {x_best}")
    print(f"Valore funzione   : f = {f_best:.6e}  (atteso: -1.0)")
    print(f"Ottimo teorico    : x = [{np.pi:.6f}, {np.pi:.6f}]")
    print(f"Rettangoli totali : {n_rects}")
