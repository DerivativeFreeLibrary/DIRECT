import unittest
import numpy as np
from direct_opt import direct_optimize_batch


class TestDirectOptimizer(unittest.TestCase):

    # ------------------------------------------------------------------
    # Funzioni di test riutilizzabili
    # ------------------------------------------------------------------

    @staticmethod
    def _sphere(X):
        """f(x) = sum(x_i^2), minimo in 0 con f=0."""
        return np.sum(X**2, axis=1)

    @staticmethod
    def _beale(X):
        """Funzione di Beale, minimo in (3, 0.5) con f=0."""
        x, y = X[:, 0], X[:, 1]
        return (
            (1.5  - x + x * y   )**2
            + (2.25 - x + x * y**2)**2
            + (2.625 - x + x * y**3)**2
        )

    # ------------------------------------------------------------------
    # Test di base
    # ------------------------------------------------------------------

    def test_sphere_function(self):
        """Verifica che l'algoritmo trovi il minimo della sfera in (0, 0)."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        x_best, f_best, _ = direct_optimize_batch(self._sphere, bounds, max_iter=50)

        self.assertLess(f_best, 1e-4)
        np.testing.assert_allclose(x_best, [0.0, 0.0], atol=1e-2)

    def test_beale_convergence(self):
        """Verifica la convergenza sulla funzione di Beale verso (3, 0.5)."""
        bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        x_best, f_best, _ = direct_optimize_batch(self._beale, bounds, max_iter=100)

        np.testing.assert_allclose(x_best, [3.0, 0.5], atol=1e-2)

    # ------------------------------------------------------------------
    # Test robustezza
    # ------------------------------------------------------------------

    def test_non_vectorized_fallback(self):
        """Verifica che il fallback funzioni con una funzione non vettorizzata
        che solleva TypeError su input 2D (il caso tipico)."""
        def sphere_strict(X):
            # Simula una funzione che rifiuta esplicitamente input 2D,
            # ad esempio una funzione scritta per un singolo punto:
            if X.ndim != 1:
                raise TypeError("Atteso array 1D, ricevuto shape: " + str(X.shape))
            return float(np.sum(X**2))

        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        x_best, f_best, _ = direct_optimize_batch(
            sphere_strict, bounds, max_iter=30
        )
        self.assertLess(f_best, 1e-3)

    def test_asymmetric_bounds(self):
        """Verifica il funzionamento con bounds non simmetrici."""
        def shifted_sphere(X):
            # Minimo in (2, 7)
            return np.sum((X - np.array([2.0, 7.0]))**2, axis=1)

        bounds = [(0.0, 5.0), (5.0, 10.0)]
        x_best, f_best, _ = direct_optimize_batch(
            shifted_sphere, bounds, max_iter=80
        )
        self.assertLess(f_best, 1e-3)
        np.testing.assert_allclose(x_best, [2.0, 7.0], atol=5e-2)

    def test_higher_dimensionality(self):
        """Verifica che l'algoritmo scala a dimensioni più alte (5D)."""
        def sphere_5d(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5.0, 5.0)] * 5
        _, f_best, _ = direct_optimize_batch(
            sphere_5d, bounds, max_iter=100, max_evals=10000
        )
        self.assertLess(f_best, 1e-2)

    def test_max_evals_respected(self):
        """Verifica che il numero di rettangoli non superi max_evals."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        max_evals = 500
        _, _, n_rects = direct_optimize_batch(
            self._sphere, bounds, max_iter=200, max_evals=max_evals
        )
        self.assertLessEqual(n_rects, max_evals)

    def test_return_types(self):
        """Verifica che i tipi di ritorno siano corretti."""
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        x_best, f_best, n_rects = direct_optimize_batch(
            self._sphere, bounds, max_iter=10
        )
        self.assertIsInstance(x_best, np.ndarray)
        self.assertIsInstance(f_best, float)
        self.assertIsInstance(n_rects, int)
        self.assertEqual(x_best.shape, (2,))


if __name__ == "__main__":
    unittest.main()
