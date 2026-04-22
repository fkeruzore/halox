import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

plt.ion()

d0 = np.load("./map.npz")["map"]
nx, ny = d0.shape
interp = RectBivariateSpline(np.arange(nx), np.arange(ny), d0)
superres = 4.0
d = interp(
    np.arange(0, nx + 1e-5, 1.0 / superres),
    np.arange(0, ny + 1e-5, 1.0 / superres),
    grid=True,
)

fig, ax = plt.subplots(figsize=(3, 3))
ax.contourf(
    d, levels=[2.00, 3.01], antialiased=True, colors=[plt.cm.bone(0.15)]
)
ax.contourf(
    d, levels=[3.00, 4.01], antialiased=True, colors=[plt.cm.bone(0.30)]
)
ax.contourf(
    d, levels=[4.00, 5.01], antialiased=True, colors=[plt.cm.bone(0.45)]
)
ax.contourf(
    d, levels=[5.00, 6.01], antialiased=True, colors=[plt.cm.bone(0.6)]
)
ax.contourf(d, levels=[6.00, 6.76], antialiased=True, colors=["#FF7E79"])
ax.contourf(d, levels=[6.75, 7.51], antialiased=True, colors=["#FFA8A5"])
ax.contourf(
    d, levels=[7.50, 8.51], antialiased=True, colors=[plt.cm.bone(0.8)]
)
ax.contourf(
    d, levels=[8.50, 99.0], antialiased=True, colors=[plt.cm.bone(1.0)]
)

ax.set_aspect("equal")
x, y, dx, dy = np.array([64.5, 64.5, 19, 19]) * superres
# ax.axvline(x, 0, 1)
# ax.axhline(y, 0, 1)
ax.set_xlim(x - dx, x + dx)
ax.set_ylim(y - dy, y + dy)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

fig.tight_layout()
fig.savefig("./logo.png", transparent=True, dpi=200)
fig.savefig("./logo.pdf", transparent=True)
