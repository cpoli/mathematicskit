"""Small supporting numerics for mathkit.calculus.systems.taylor_series:
partial-sum sequences and empirical truncation error, used by the
visualizers and examples to show a series actually converging (or
diverging outside its radius)."""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["partial_sums", "truncation_error"]


def partial_sums(coefficients: np.ndarray, x: float) -> np.ndarray:
    r"""Partial sums :math:`S_k(x) = \sum_{n=0}^{k} a_n x^n` for every ``k``.

    Parameters
    ----------
    coefficients : ndarray, shape (order + 1,)
    x : float

    Returns
    -------
    ndarray, shape (order + 1,)
        ``result[k]`` is the degree-``k`` partial sum.

    Examples
    --------
    >>> import numpy as np
    >>> sums = partial_sums(np.array([1.0, 1.0, 0.5, 1.0 / 6.0]), 1.0)
    >>> np.round(sums, 6)
    array([1.      , 2.      , 2.5     , 2.666667])
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    terms = coefficients * x ** np.arange(coefficients.shape[0])
    return np.cumsum(terms)


def truncation_error(f: Callable[[float], float], coefficients: np.ndarray, x: float) -> np.ndarray:
    r"""Empirical truncation error ``|f(x) - S_k(x)|`` for every partial sum.

    Parameters
    ----------
    f : callable
        The function the series is meant to approximate.
    coefficients : ndarray, shape (order + 1,)
    x : float

    Returns
    -------
    ndarray, shape (order + 1,)

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.calculus.systems.taylor_series import maclaurin_coefficients
    >>> coeffs = maclaurin_coefficients("exp", 15)
    >>> errors = truncation_error(np.exp, coeffs, 1.0)
    >>> bool(errors[-1] < errors[0])
    True
    """
    sums = partial_sums(coefficients, x)
    return np.abs(f(x) - sums)
