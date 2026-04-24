import numpy as np
from direct_opt import direct_optimize_batch


def beale(X):
    """
    Funzione di Beale vettorializzata.

    Minimo globale: f(3.0, 0.5) = 0.0
    Dominio standard: x, y in [-4.5, 4.5]

    Parametri:
    ----------
    X : ndarray di shape (n_points, 2)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    x = X[:, 0]
    y = X[:, 1]
    return (
        (1.5   - x + x * y   )**2
        + (2.25  - x + x * y**2)**2
        + (2.625 - x + x * y**3)**2
    )


if __name__ == "__main__":
    bounds = [(-4.5, 4.5), (-4.5, 4.5)]

    x_best, f_best, n_rects = direct_optimize_batch(
        beale,
        bounds,
        max_iter=100
    )

    print(f"Minimo trovato    : x = {x_best}")
    print(f"Valore funzione   : f = {f_best:.6e}  (atteso: 0.0)")
    print(f"Rettangoli totali : {n_rects}")
