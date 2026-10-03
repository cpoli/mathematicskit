"""Plotting helpers for mathematicskit.information_theory: the binary tree of
a prefix code, and the information diagram of two random variables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from mathematicskit.information_theory.systems.entropy import conditional_entropy, entropy, mutual_information

__all__ = ["plot_code_tree", "plot_information_diagram"]


def plot_code_tree(code, ax=None):
    """Draw a prefix code as a binary tree: left edges are ``0``, right edges ``1``, leaves are symbols.

    Parameters
    ----------
    code : PrefixCode
        E.g. from :func:`~mathematicskit.information_theory.systems.source_coding.huffman_code`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    leaves = sorted(code.codewords.items(), key=lambda sw: sw[1])
    x_of: dict = {}
    for i, (_, word) in enumerate(leaves):
        x_of[word] = float(i)
    for depth in range(max(len(w) for _, w in leaves) - 1, -1, -1):
        prefixes = {w[:depth] for _, w in leaves if len(w) > depth}
        for prefix in prefixes:
            children = [x_of[prefix + b] for b in "01" if prefix + b in x_of]
            x_of[prefix] = sum(children) / len(children)
    for node, x in x_of.items():
        if node:
            parent = node[:-1]
            ax.plot([x_of[parent], x], [-len(parent), -len(node)], color="0.5", zorder=1)
            ax.annotate(node[-1], ((x_of[parent] + x) / 2, -len(node) + 0.5), fontsize=8, color="0.3", ha="center")
    for symbol, word in leaves:
        p = code.probabilities.get(symbol, 0.0)
        ax.scatter(x_of[word], -len(word), s=300, color="tab:blue", zorder=2)
        ax.annotate(f"{symbol}\n{p:.2g}", (x_of[word], -len(word) - 0.35), ha="center", va="top", fontsize=8)
    internal = [w for w in x_of if w not in code.codewords.values()]
    ax.scatter([x_of[w] for w in internal], [-len(w) for w in internal], s=40, color="0.3", zorder=2)
    ax.set_title(f"average length {code.average_length:.3f} bits, entropy {code.entropy:.3f} bits")
    ax.axis("off")
    ax.margins(0.1, 0.2)
    return ax


def plot_information_diagram(joint, ax=None):
    r"""Two overlapping circles for :math:`H(X)` and :math:`H(Y)`, labelled :math:`H(X|Y)`, :math:`I(X;Y)` and :math:`H(Y|X)`.

    The circle overlap is drawn at a fixed size; the labels carry the
    values.

    Parameters
    ----------
    joint : array_like, shape (m, n)
        ``joint[i, j] = P(X=i, Y=j)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    joint = np.asarray(joint, dtype=float)
    h_y_given_x = conditional_entropy(joint)
    h_x_given_y = conditional_entropy(joint.T)
    info = mutual_information(joint)
    ax.add_patch(Circle((-0.5, 0), 1.0, alpha=0.35, color="tab:blue"))
    ax.add_patch(Circle((0.5, 0), 1.0, alpha=0.35, color="tab:orange"))
    ax.text(-0.95, 0, f"H(X|Y)\n{h_x_given_y:.3f}", ha="center", va="center")
    ax.text(0, 0, f"I(X;Y)\n{info:.3f}", ha="center", va="center", fontweight="bold")
    ax.text(0.95, 0, f"H(Y|X)\n{h_y_given_x:.3f}", ha="center", va="center")
    ax.text(-0.5, 1.1, f"H(X) = {entropy(joint.sum(axis=1)):.3f}", ha="center")
    ax.text(0.5, -1.2, f"H(Y) = {entropy(joint.sum(axis=0)):.3f}", ha="center")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"H(X,Y) = {entropy(joint):.3f} bits")
    return ax
