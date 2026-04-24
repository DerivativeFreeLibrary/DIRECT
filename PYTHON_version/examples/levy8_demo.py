import numpy as np
from direct_opt import direct_optimize_batch


def levy8(X):
    """
    Funzione di Levy (variante 8) vettorializzata.

    Minimo globale: f(1, ..., 1) = 0.0
    Dominio standard: x_i in [-10, 10]

    La funzione e' definita come:
        f(x) = (pi/n) * [ 10*sin(pi*x_0)^2
                         + sum_{i=0}^{n-2} (x_i - 1)^2 * (1 + 10*sin(pi*x_{i+1})^2)
                         + (x_{n-1} - 1)^2 ]

    Parametri:
    ----------
    X : ndarray di shape (n_points, n_dims)

    Returns:
    --------
    ndarray di shape (n_points,)
    """
    n_dims = X.shape[1]

    # Termine iniziale
    f = 10.0 * np.sin(np.pi * X[:, 0])**2

    # Termini intermedi: sommatoria da i=0 a n-2
    # (x_i - 1)^2 * (1 + 10 * sin(pi * x_{i+1})^2)
    term_i    = (X[:, :-1] - 1.0)**2
    term_next = 1.0 + 10.0 * np.sin(np.pi * X[:, 1:])**2
    f += np.sum(term_i * term_next, axis=1)

    # Termine finale
    f += (X[:, -1] - 1.0)**2

    return f * (np.pi / n_dims)


if __name__ == "__main__":
    n_dims = 3
    bounds = [(-10.0, 10.0)] * n_dims

    x_best, f_best, n_rects = direct_optimize_batch(
        levy8,
        bounds,
        max_iter=100,
        max_evals=20000
    )

    print(f"Minimo trovato    : x = {x_best}")
    print(f"Valore funzione   : f = {f_best:.6e}  (atteso: 0.0)")
    print(f"Ottimo teorico    : x = {[1.0] * n_dims}")
    print(f"Rettangoli totali : {n_rects}")
