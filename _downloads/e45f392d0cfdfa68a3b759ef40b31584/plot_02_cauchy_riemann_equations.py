r"""
The Cauchy-Riemann equations: u_x = v_y, u_y = -v_x
===================================================

A function :math:`f = u + iv` is complex differentiable exactly where
its real and imaginary parts satisfy the Cauchy-Riemann equations. The
residual :math:`\max(|u_x - v_y|, |u_y + v_x|)` vanishes everywhere for
:math:`z^2` and :math:`e^z`, but not for :math:`\bar z`,
:math:`|z|^2`, or :math:`\operatorname{Re} z`. :math:`|z|^2` is the
interesting case: it satisfies the equations only at :math:`z = 0`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import cauchy_riemann, complex_derivative, complex_grid

# %%
# Holomorphic vs. non-holomorphic functions
# -----------------------------------------

functions = {
    "z^2": lambda z: z**2,
    "exp(z)": np.exp,
    "conj(z)": np.conj,
    "|z|^2": lambda z: abs(z) ** 2,
    "Re(z)": lambda z: z.real,
}
z0 = 0.7 - 0.4j
for name, f in functions.items():
    r = cauchy_riemann(f, z0)
    print(f"{name:8s} u_x={r.u_x:+.3f} v_y={r.v_y:+.3f} u_y={r.u_y:+.3f} v_x={r.v_x:+.3f}  residual={r.residual:.1e}")
print("d/dz z^2 at z0:", complex_derivative(lambda z: z**2, z0), "= 2 z0 =", 2 * z0)

# %%
# Where does the squared modulus satisfy the equations?
# -----------------------------------------------------

z = complex_grid((-1, 1), (-1, 1), 41)
residual = np.vectorize(lambda p: cauchy_riemann(lambda w: abs(w) ** 2, p).residual)(z)
fig, ax = plt.subplots()
image = ax.imshow(residual, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
fig.colorbar(image, ax=ax, label="Cauchy-Riemann residual")
ax.set_title(r"$|z|^2$ is complex differentiable only at $z = 0$")
ax.set_xlabel("Re z")
ax.set_ylabel("Im z")
