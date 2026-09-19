"""Smoke tests for mathematicskit.number_theory.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathematicskit.number_theory.systems.continued_fractions import continued_fraction_expansion
from mathematicskit.number_theory.visualizers.plots import plot_convergent_errors, plot_prime_counting


def test_plot_prime_counting_returns_axes():
    ax = plot_prime_counting(200)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_convergent_errors_returns_axes():
    result = continued_fraction_expansion(3.14159265358979, max_terms=8)
    ax = plot_convergent_errors(result, x=3.14159265358979)
    assert isinstance(ax, matplotlib.axes.Axes)
