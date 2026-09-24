"""Least-squares polynomial regression, via :func:`numpy.linalg.lstsq`.

Fits a degree-``d`` polynomial minimizing :math:`\\sum_i (y_i -
p(x_i))^2` for the Vandermonde design matrix :math:`V`, the problem
posed in Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 8.1
("Discrete Least Squares Approximation"). Rather than forming and
solving the normal equations :math:`V^T V c = V^T y` by hand (which
squares :math:`V`'s condition number -- see :mod:`mathematicskit.linalg.systems.stability`
for a worked-out demonstration of exactly this instability),
:func:`numpy.linalg.lstsq` solves the least-squares problem directly on
:math:`V` via an SVD-based routine (LAPACK's ``?gelsd``), which is
numerically stable even when :math:`V` is ill-conditioned. The condition
number reported in :class:`~mathematicskit.numerical_analysis.core.base.RegressionResult`
comes from :func:`~mathematicskit.numerical_analysis.utils.error_analysis.condition_number`
(:func:`numpy.linalg.cond`) applied to :math:`V` itself.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.numerical_analysis.core.base import RegressionResult
from mathematicskit.numerical_analysis.utils.error_analysis import condition_number

__all__ = ["PolynomialRegression"]


class PolynomialRegression:
    r"""Least-squares polynomial fit of degree `degree` to ``(x_i, y_i)``.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Data points.
    degree : int
        Polynomial degree (``degree < n`` is required so the fit isn't
        underdetermined into exact interpolation).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(-1.0, 1.0, 50)
    >>> y = 3.0 * x**2 - 2.0 * x + 1.0  # noiseless: fit should be essentially exact
    >>> model = PolynomialRegression(x, y, degree=2)
    >>> result = model.fit()
    >>> np.allclose(result.coefficients, [3.0, -2.0, 1.0], atol=1e-8)
    True
    >>> round(result.r_squared, 6)
    1.0
    """

    def __init__(self, x, y, degree: int):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if degree >= x.shape[0]:
            raise ValueError("degree must be smaller than the number of data points")
        self.x = x
        self.y = y
        self.degree = degree

    def _vandermonde(self) -> np.ndarray:
        # Columns x^degree, x^(degree-1), ..., x^0 (numpy.polyval convention).
        return np.vander(self.x, self.degree + 1)

    def fit(self) -> RegressionResult:
        """Fit the model and return the result.

        Returns
        -------
        RegressionResult
        """
        v = self._vandermonde()
        coefficients, _residuals, _rank, _sv = np.linalg.lstsq(v, self.y, rcond=None)

        fitted = v @ coefficients
        residuals = self.y - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((self.y - np.mean(self.y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        n = self.x.shape[0]
        p = self.degree + 1
        if n - p > 0 and ss_tot > 0:
            adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p)
        else:
            adjusted = r_squared

        try:
            cond = condition_number(v)
        except np.linalg.LinAlgError:
            cond = None

        return RegressionResult(
            coefficients=coefficients,
            fitted_values=fitted,
            residuals=residuals,
            r_squared=r_squared,
            adjusted_r_squared=adjusted,
            condition_number=cond,
        )

    def predict(self, x_new) -> np.ndarray:
        """Evaluate the fitted polynomial at new points.

        Fits the model on first use and caches the coefficients, so calling
        :meth:`fit` beforehand is optional; the cache is never invalidated,
        since a :class:`PolynomialRegression` instance's data and degree are
        fixed at construction.

        Parameters
        ----------
        x_new : array-like of float

        Returns
        -------
        ndarray
        """
        if not hasattr(self, "_coefficients"):
            self._coefficients = self.fit().coefficients
        return np.polyval(self._coefficients, np.asarray(x_new, dtype=np.float64))
