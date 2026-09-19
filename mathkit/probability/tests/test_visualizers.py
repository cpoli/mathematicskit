"""Smoke tests for mathkit.probability.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathkit.probability.systems.continuous import Normal
from mathkit.probability.systems.discrete import Poisson
from mathkit.probability.systems.limit_theorems import central_limit_theorem_sample_means
from mathkit.probability.systems.markov_chain import MarkovChain
from mathkit.probability.visualizers.plots import plot_clt_histogram, plot_distribution, plot_transition_matrix


def test_plot_distribution_discrete_returns_axes():
    ax = plot_distribution(Poisson(mu=4.0))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_distribution_continuous_returns_axes():
    ax = plot_distribution(Normal(mu=0.0, sigma=1.0))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_clt_histogram_returns_axes():
    z = central_limit_theorem_sample_means(Poisson(mu=3.0), n=100, n_trials=500, seed=0)
    ax = plot_clt_histogram(z)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_transition_matrix_returns_axes():
    chain = MarkovChain(np.array([[0.9, 0.1], [0.5, 0.5]]))
    ax = plot_transition_matrix(chain)
    assert isinstance(ax, matplotlib.axes.Axes)
