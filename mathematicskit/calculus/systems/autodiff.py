r"""Reverse-mode (backpropagation-style) automatic differentiation.

Builds a dynamic computation graph of :class:`Variable` nodes as a
function executes, each remembering its parents and the local partial
derivative(s) needed to propagate an incoming gradient to them (the
"vector-Jacobian product" of one operation); calling :meth:`Variable.backward`
then walks the graph in reverse topological order, accumulating
:math:`\partial \text{output}/\partial v` at every node ``v`` via the
multivariate chain rule -- exactly the algorithm behind modern deep
learning frameworks, in miniature. See Rumelhart, Hinton & Williams
(1986), *Learning representations by back-propagating errors*, Nature
323, and Griewank & Walther, *Evaluating Derivatives*, 2nd ed., Ch. 3-4
(the "reverse mode" / "adjoint" method).
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["Variable", "gradient"]


class Variable:
    r"""A scalar node in a reverse-mode autodiff computation graph.

    Parameters
    ----------
    value : float
        The node's forward value.
    _parents : list of (Variable, float), optional
        Internal: ``(parent, local_partial_derivative)`` pairs recorded
        by the operation that created this node.

    Examples
    --------
    >>> x = Variable(3.0)
    >>> y = x * x + 2.0 * x  # f(x) = x^2 + 2x
    >>> y.backward()
    >>> round(x.grad, 10)  # f'(3) = 2*3 + 2 = 8
    8.0
    """

    __slots__ = ("value", "grad", "_parents")

    def __init__(self, value: float, _parents: list[tuple[Variable, float]] = None):
        self.value = float(value)
        self.grad = 0.0
        self._parents = _parents or []

    def __repr__(self):
        return f"Variable({self.value!r}, grad={self.grad!r})"

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Variable) else Variable(other)

    def __add__(self, other):
        other = self._coerce(other)
        return Variable(self.value + other.value, [(self, 1.0), (other, 1.0)])

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        return Variable(self.value - other.value, [(self, 1.0), (other, -1.0)])

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return Variable(self.value * other.value, [(self, other.value), (other, self.value)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        return Variable(self.value / other.value, [(self, 1.0 / other.value), (other, -self.value / other.value**2)])

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __neg__(self):
        return Variable(-self.value, [(self, -1.0)])

    def __pow__(self, p: float):
        return Variable(self.value**p, [(self, p * self.value ** (p - 1))])

    def sin(self):
        return Variable(math.sin(self.value), [(self, math.cos(self.value))])

    def cos(self):
        return Variable(math.cos(self.value), [(self, -math.sin(self.value))])

    def exp(self):
        e = math.exp(self.value)
        return Variable(e, [(self, e)])

    def log(self):
        return Variable(math.log(self.value), [(self, 1.0 / self.value)])

    def _topological_order(self) -> list[Variable]:
        order: list[Variable] = []
        visited = set()

        def visit(node: Variable):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent, _ in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(self):
        """Propagate gradients back through the computation graph.

        Sets ``self.grad = 1`` (the seed) and accumulates
        :math:`\\partial \\text{self}/\\partial v` into ``v.grad`` for
        every node ``v`` reachable as a parent, via reverse topological
        traversal.
        """
        for node in self._topological_order():
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(self._topological_order()):
            for parent, local_partial in node._parents:
                parent.grad += node.grad * local_partial


def gradient(f: Callable[..., Variable], x: list[float]) -> list[float]:
    r"""Gradient of a scalar function of several variables via reverse-mode AD.

    Parameters
    ----------
    f : callable
        ``f(*variables) -> Variable``, built from :class:`Variable`
        arithmetic/methods.
    x : list of float
        Point at which to evaluate the gradient.

    Returns
    -------
    list of float
        :math:`\nabla f(x)`.

    Examples
    --------
    >>> f = lambda x, y: x * x * y + y  # df/dx = 2xy, df/dy = x^2 + 1
    >>> grad = gradient(f, [3.0, 2.0])
    >>> [round(g, 10) for g in grad]
    [12.0, 10.0]
    """
    variables = [Variable(xi) for xi in x]
    out = f(*variables)
    out.backward()
    return [v.grad for v in variables]
