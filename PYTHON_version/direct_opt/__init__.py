"""
direct_opt — Libreria Python per l'ottimizzazione globale con il metodo DIRECT.

Espone la funzione principale:
    direct_optimize_batch : implementazione batch-vettorizzata dell'algoritmo DIRECT.
"""

from .core import direct_optimize_batch

__version__ = "0.1.0"
__all__ = ["direct_optimize_batch"]