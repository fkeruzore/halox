"""Plot HMF benchmark results as stacked horizontal bars.

Speedup is relative to the CPU JIT Analytical baseline.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.style.use(["seaborn-v0_8-darkgrid", "petroff10"])
plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

df = pd.read_csv("benchmark_hmf_results.csv")
baseline = df.loc[df["name"] == "Analytical; CPU; JIT", "time_s"].values[0]
df["speedup"] = baseline / df["time_s"]

colossus_time = df.loc[df["name"] == "colossus", "time_s"].values[0]
colossus_speedup = baseline / colossus_time

df_halox = df[df["name"].str.contains("; ")].copy()
df_halox[["method", "device", "jit"]] = df_halox["name"].str.split(
    "; ", expand=True
)
df_halox["label"] = df_halox["method"] + "; " + df_halox["device"]

pivot = df_halox.pivot(index="label", columns="jit", values="speedup")
pivot = pivot.loc[
    ["Analytical; CPU", "Analytical; GPU", "Emulator; CPU", "Emulator; GPU"]
]

colors = ["C0" if "CPU" in lbl else "C1" for lbl in pivot.index]
hatches = [r"xx" if "Analytical" in lbl else r"//" for lbl in pivot.index]

fig, ax = plt.subplots(figsize=(8, 4))

# Full bar: JIT speedup (transparent)
ax.barh(
    pivot.index,
    pivot["JIT"],
    color=colors,
    hatch=hatches,
    edgecolor="white",
    alpha=0.35,
)
# Overlapping bar: No JIT speedup (opaque)
ax.barh(
    pivot.index,
    pivot["No JIT"],
    color=colors,
    hatch=hatches,
    edgecolor="white",
    alpha=1.0,
)

for i, (_, row) in enumerate(pivot.iterrows()):
    ax.text(
        row["JIT"] * 1.08,
        i,
        f"{row['JIT']:.3g}×",
        va="center",
        ha="left",
        color="k",
        fontsize=10,
    )
    ax.text(
        row["No JIT"] * 1.08,
        i,
        f"{row['No JIT']:.3g}×",
        va="center",
        ha="left",
        color="k",
        fontsize=10,
    )

ax.barh(
    ["Colossus (CPU)"],
    [colossus_speedup],
    color="C3",
    edgecolor="white",
    alpha=1.0,
)
ax.text(
    colossus_speedup * 1.08,
    "Colossus (CPU)",
    f"{colossus_speedup:.3g}×",
    va="center",
    ha="left",
    color="k",
    fontsize=10,
)

ax.axvline(1, color="k", ls="--", lw=0.8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Speedup vs. CPU JIT Analytical (log scale; higher = better)")

legend_handles = [
    Patch(facecolor="C0", edgecolor="white", label="CPU"),
    Patch(facecolor="C1", edgecolor="white", label="GPU"),
    Patch(
        facecolor="grey", hatch=r"xx", edgecolor="white", label="Analytical"
    ),
    Patch(facecolor="grey", hatch=r"//", edgecolor="white", label="Emulator"),
    Patch(facecolor="grey", alpha=1.0, edgecolor="white", label="No JIT"),
    Patch(facecolor="grey", alpha=0.35, edgecolor="white", label="JIT"),
    plt.Line2D([0], [0], color="k", ls="--", lw=0.8, label="Baseline"),
]
ax.legend(handles=legend_handles, handlelength=2.5, handleheight=1.5)

xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, xmax * 4)
fig.tight_layout()
fig.savefig("../benchmark_hmf_results.png", dpi=500)
plt.show()
