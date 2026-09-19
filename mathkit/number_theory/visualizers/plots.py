"""Plotting helpers for mathkit.number_theory: prime distribution and
convergent-approximation-error plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_prime_counting", "plot_convergent_errors"]


def plot_prime_counting(limit: int, ax=None):
    """Plot the prime-counting function ``pi(n)`` up to `limit`.

    Parameters
    ----------
    limit : int
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from mathkit.number_theory.systems.primality import sieve_of_eratosthenes

    if ax is None:
        _, ax = plt.subplots()
    primes = sieve_of_eratosthenes(limit)
    ns = np.arange(1, limit + 1)
    pi_n = np.searchsorted(primes, ns, side="right")
    ax.plot(ns, pi_n, label="pi(n)")
    ax.plot(ns, ns / np.log(np.maximum(ns, 2)), "--", label="n / ln(n)", color="firebrick")
    ax.set_xlabel("n")
    ax.set_ylabel("number of primes <= n")
    ax.set_title("Prime-counting function")
    ax.legend()
    return ax


def plot_convergent_errors(result, x: float, ax=None):
    """Semilog plot of a continued fraction's convergent approximation error vs. denominator.

    Parameters
    ----------
    result : ContinuedFractionResult
        From :func:`~mathkit.number_theory.systems.continued_fractions.continued_fraction_expansion`.
    x : float
        The value being approximated.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    denominators = np.array([q for _, q in result.convergents])
    errors = np.array([abs(p / q - x) for p, q in result.convergents])
    errors = np.where(errors == 0.0, np.nan, errors)
    ax.semilogy(denominators, errors, "o-")
    ax.set_xlabel("denominator q")
    ax.set_ylabel("|p/q - x|")
    ax.set_title("Continued-fraction convergent error")
    return ax
