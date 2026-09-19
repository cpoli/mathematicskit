"""Mathematical constants and shared numerical tolerances used across mathkit.

Mathematics has no unit system to convert between (unlike physicskit's
SI-value table or chemistrykit's periodic-table data), so this module is
deliberately small: a single source of truth for a handful of named
mathematical constants, plus the default convergence tolerances used by
iterative/root-finding algorithms across every domain so that "did this
converge" means the same thing everywhere in mathkit instead of each
module inventing its own magic number.

Values for irrational constants are taken from :mod:`numpy` where it
already defines them (``PI``, ``E``), and computed here to full
``float64`` precision otherwise (``GOLDEN_RATIO``, ``EULER_MASCHERONI``,
``SQRT2``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PI",
    "E",
    "GOLDEN_RATIO",
    "EULER_MASCHERONI",
    "SQRT2",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "DEFAULT_MAX_ITER",
]

#: Archimedes' constant, :math:`\pi`.
PI = np.pi

#: Euler's number, :math:`e`.
E = np.e

#: The golden ratio, :math:`\varphi = (1 + \sqrt{5}) / 2`.
GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

#: The Euler-Mascheroni constant, :math:`\gamma = \lim_{n\to\infty}\left(\sum_{k=1}^{n} 1/k - \ln n\right)`,
#: to float64 precision (OEIS A001620).
EULER_MASCHERONI = 0.5772156649015328606065120900824024310421593359399235988057672348849

#: :math:`\sqrt{2}`.
SQRT2 = np.sqrt(2.0)

#: Default absolute tolerance for iterative/root-finding convergence checks.
DEFAULT_ATOL = 1e-10

#: Default relative tolerance for iterative/root-finding convergence checks.
DEFAULT_RTOL = 1e-8

#: Default maximum iteration count for iterative solvers/root finders.
DEFAULT_MAX_ITER = 1000
