"""Matplotlib figures for the report."""
import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def draw_obstacles(ax, obstacles):
    for o in obstacles:
        ax.plot_surface(*o.surface(40, 20), color="#b07c7c", alpha=0.55, linewidth=0, zorder=1)
        ax.plot_wireframe(*o.surface(24, 12, inflated=True), color="k", alpha=0.12, linewidth=0.5)


def set_equal(ax, lims):
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect([lims[i][1] - lims[i][0] for i in range(3)])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")


VIEWS = (("isometric view", 22, -125), ("top view (x-y)", 90, -90), ("side view (y-z)", 0, 0))


def plot_scene_3d(scene, results, nominal=None, fname=None, views=VIEWS):
    """Obstacle, attractor and trajectories, seen from two viewpoints."""
    fig = plt.figure(figsize=(18, 6))
    att = scene["ds"].attractor
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        draw_obstacles(ax, scene["obstacles"])
        for j, (label, r) in enumerate(results.items()):
            c = COLORS[j % len(COLORS)]
            tr = r["traj"]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=2.2, label=label)
            ax.scatter(*tr[0], color=c, marker="o", s=30, edgecolor="k")
            if nominal is not None:
                tn = nominal[label]["traj"]
                ax.plot(tn[:, 0], tn[:, 1], tn[:, 2], color=c, lw=1, ls=":", alpha=0.7)
        ax.scatter(*att, marker="*", s=350, color="gold", edgecolor="k", zorder=10, label="attractor x*")
        set_equal(ax, scene["lims"])
        ax.view_init(elev=elev, azim=azim)
        if elev == 90:                       # top view: z axis is meaningless
            ax.set_zticks([]); ax.set_zlabel("")
        if elev == 0 and azim == 0:          # side view: x axis points at the camera
            ax.set_xticks([]); ax.set_xlabel("")
        ax.set_title(title)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle(scene["name"] + ("   (dotted: nominal DS without modulation)" if nominal else ""), y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig


def plot_lyapunov_and_distance(scene, results, nominal=None, fname=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for j, (label, r) in enumerate(results.items()):
        c = COLORS[j % len(COLORS)]
        m = r["metrics"]
        axs[0].semilogy(r["t"], m["V"] + 1e-12, color=c, label=label)
        if nominal is not None:
            axs[0].semilogy(nominal[label]["t"], nominal[label]["metrics"]["V"] + 1e-12, color=c, ls=":", alpha=0.7)
        axs[1].plot(r["t"], m["gamma"], color=c)
        axs[2].plot(r["t"], 100 * m["distance"], color=c)
    axs[0].set(title=r"Lyapunov $V=\frac{1}{2}\|x-x^*\|^2$ (dotted: nominal)", xlabel="t [s]", ylabel="V")
    axs[1].axhline(1, color="r", ls="--", lw=1)
    axs[1].set(title=r"$\Gamma(x(t))$  (must stay $\geq 1$)", xlabel="t [s]", ylim=(0, 8))
    axs[2].axhline(0, color="r", ls="--", lw=1)
    axs[2].set(title="distance to real obstacle surface", xlabel="t [s]", ylabel="[cm]")
    axs[0].legend(fontsize=8)
    fig.suptitle(scene["name"])
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig


def plot_modulation_field(scene, plane_z=None, fname=None, n=25):
    """Quiver plot of nominal vs modulated velocity in the horizontal plane through the obstacle."""
    from .modulation import modulated_velocity
    o = scene["obstacles"][0]
    z = o.c[2] if plane_z is None else plane_z
    (x0, x1), (y0, y1), _ = scene["lims"]
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for k, mod in enumerate([False, True]):
        U = np.zeros_like(X); V = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                p = np.array([X[i, j], Y[i, j], z])
                if o.gamma(p) < 1:
                    U[i, j] = V[i, j] = np.nan
                    continue
                v = modulated_velocity(p, scene["ds"], scene["obstacles"]) if mod else scene["ds"](p)
                U[i, j], V[i, j] = v[0], v[1]
        ax = axs[k]
        ax.streamplot(X, Y, U, V, color=np.hypot(U, V), cmap="viridis", density=1.3)
        # cross-section of the obstacle at height z (points of a dense surface mesh)
        Xs, Ys, Zs = o.surface(200, 200)
        mask = np.abs(Zs - z) < 0.004
        ax.plot(Xs[mask], Ys[mask], ".", color="#b07c7c", ms=2)
        ax.plot(*scene["ds"].attractor[:2], "*", color="gold", mec="k", ms=18)
        ax.set(aspect="equal", xlim=(x0, x1), ylim=(y0, y1), xlabel="x [m]", ylabel="y [m]",
               title=("modulated  M(x) f(x)" if mod else "nominal  f(x)") + f"   (slice z = {z:.2f} m)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig
