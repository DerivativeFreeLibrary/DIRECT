# DIRECT-Optimizer

A high-performance, vectorized implementation of the **DIRECT** (DIviding RECTangles) algorithm for derivative-free global optimization in Python.

This library leverages **NumPy** for batch evaluations and a **Structure of Arrays (SoA)** memory layout, making it significantly faster than traditional pure-Python implementations for expensive objective functions.

## Key Features

* **Batch Evaluation & Scalar Fallback:** Evaluates the objective function on multiple points simultaneously, allowing users to exploit vectorization and multi-core CPUs. If the provided function is not vectorized, the solver automatically applies a scalar fallback.
* **Structure of Arrays (SoA):** Uses flat `float64` arrays for rectangle data, improving cache locality, reducing memory overhead compared to object-oriented trees, and supporting deep geometric subdivisions.
* **Geometric Robustness:** Implements a numerical safety filter (ignoring rectangles with linear dimensions below 10^-8) to prevent "geometric collapse" and avoid wasting computational budget in regions dominated by round-off errors.
* **Continuous Local Refinement:** The algorithm features a specific architectural choice where the rectangle containing the current global minimum is always subdivided. This ensures continuous refinement of the best-known solution, while the epsilon parameter strictly regulates the exploration of sub-optimal regions.
* **Robust Input Validation:** Comprehensive checks on bounds and objective function outputs to ensure solver stability.
* **Pure Python/NumPy:** No compilation required, easy to install and modify.

## Installation

To install the library in editable mode (recommended for development):

```bash
cd DIRECT_optimizer
pip install -e .
```

(Works on Linux, macOS, and Windows)

## Quick start

Here is a simple example minimizing the Beale function:

```python
import numpy as np
from direct_opt import direct_optimize_batch

def beale_vectorized(X):
    """
    Vectorized objective function.
    Input shape: (n_points, 2)
    Output shape: (n_points,)
    """
    x = X[:, 0]
    y = X[:, 1]
    
    # Beale Function formula
    f = (1.5 - x + x*y)**2 + \
        (2.25 - x + x*y**2)**2 + \
        (2.625 - x + x*y**3)**2
    
    return f

# Define bounds: x in [-4.5, 4.5], y in [-4.5, 4.5]
bounds = [(-4.5, 4.5), (-4.5, 4.5)]

# Run Optimization
x_best, f_best, n_rects = direct_optimize_batch(
    func=beale_vectorized,
    bounds=bounds,
    max_iter=50,
    max_evals=10000
)

print(f"Global Minimum found at: {x_best}")
print(f"Function Value: {f_best}")
print(f"Total Evaluations: {n_rects}")
```

## Theory

The algorithm implements the DIRECT method described by Jones, Perttunen, and Stuckman (1993). The name stands for **DIviding RECTangles**: the algorithm normalizes the search space into a unit hypercube and iteratively subdivides it into smaller hyper-rectangles (using trisection), evaluating the objective function at their centers.

At each iteration, **potentially optimal rectangles (POH)** are selected via the Lower Convex Hull of the (diameter, function-value) plot using a Monotone Chain approach. A rectangle is potentially optimal if it lies on the lower convex hull and satisfies an epsilon-condition that ensures a meaningful improvement over the current best value is geometrically possible. 

Unlike standard implementations, this library always classifies the rectangle containing the current global minimum as a POH, bypassing the epsilon filter. This derogation guarantees that the algorithm systematically refines the local optimum found, delegating to the epsilon parameter solely the role of regulating exploration in sub-optimal regions.

Furthermore, this implementation introduces a **vectorized batch strategy**: all new centers generated in a single iteration are evaluated in one NumPy call, minimizing Python interpreter overhead and allowing users to exploit hardware-level parallelism (e.g., SIMD, multi-threaded BLAS) simply by providing a vectorized objective function.

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE] file for details.

This implementation is inspired by the C implementation of DIRECT by *G. Liuzzi, S. Lucidi, and V. Piccialli*, which is also licensed under GPLv3.

## Running Tests

```bash
python -m pytest tests/ -v
```

(Works on Linux, macOS, and Windows)

## References

Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993). *Lipschitzian optimization without the Lipschitz constant*. Journal of Optimization Theory and Applications, 79(1), 157–181.

---
*Developed as part of a Bachelor's Thesis in Computer Engineering.*