r"""
Reverse-mode automatic differentiation for multivariable gradients
=========================================================================

A minimal backpropagation-style engine: one forward pass builds a
computation graph, then one backward pass computes the gradient with
respect to every input in a single traversal -- the technique underlying
modern deep-learning frameworks, applied here to a small analytic
function.
"""

# %%
import math

from mathematicskit.calculus.systems.autodiff import Variable, gradient

# %%
# Gradient of a simple multivariable function
# ---------------------------------------------------

f = lambda x, y: x * x * y + y.sin() * x
x0, y0 = 2.0, 0.5
grad = gradient(f, [x0, y0])
print(f"gradient at ({x0}, {y0}): {grad}")

# Closed-form check: df/dx = 2xy + sin(y), df/dy = x^2 + x*cos(y)
expected_dx = 2.0 * x0 * y0 + math.sin(y0)
expected_dy = x0**2 + x0 * math.cos(y0)
print(f"closed-form gradient:      [{expected_dx}, {expected_dy}]")

# %%
# Shared subexpressions accumulate gradient contributions correctly
# ------------------------------------------------------------------------

x = Variable(3.0)
y = x * x + x  # used twice: y = x^2 + x, dy/dx = 2x + 1
y.backward()
print(f"d(x^2 + x)/dx at x=3: {x.grad} (expected {2 * 3.0 + 1})")
