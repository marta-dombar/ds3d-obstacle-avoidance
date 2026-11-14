"""
Part 1 (workflow steps 1-3): the Lab 1 DS ported to Python and plugged into the
2D obstacle-avoidance package of Lab 2 (patched copy in ./exercise2_obstacle).

Uses lpv_model.mat (exported from MATLAB with matlab/export_lpv_model.m) if it
exists, otherwise a synthetic LPV-DS with the same structure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_rk4, obs_avoidance_interpolation_moving

from ds3d.dynamics import LPVDS


# Plot ranges, start points and obstacle for each model.
# Edit CONFIG["learned"] if you re-learn the DS with different demonstrations.
CONFIG = {
    "learned": dict(x_range=[-1.9, 0.3], y_range=[-0.35, 0.95],
                    starts=[[-1.75, 0.75], [-1.5, 0.45], [-1.2, -0.05], [-1.75, 0.15], [0.2, 0.0]],
                    obstacle_center=(-0.6, 0.30), obstacle_axes=(0.10, 0.07), th_r=0.2, sf=1.1),
    "synthetic": dict(x_range=[-2.5, 1.5], y_range=[-1.5, 1.5],
                      starts=[[-2.3, -1.2], [-2.3, 1.2], [-2.0, 0.2], [1.3, 1.3], [1.3, -1.3]],
                      obstacle_center=(-1.3, 0.8), obstacle_axes=(0.3, 0.2), th_r=0.4, sf=1.1),
}


def load_ds(mat_path="lpv_model.mat"):
    if os.path.exists(mat_path):
        print(f"[2D] Loading learned LPV-DS from {mat_path}")
        return LPVDS.from_mat(mat_path), "learned LPV-DS (Lab 1)", "learned"
    print("[2D] lpv_model.mat not found -> using the synthetic LPV-DS")
    return LPVDS.synthetic(), "synthetic LPV-DS", "synthetic"


def run(out_dir="figures", show=False, mat_path="lpv_model.mat"):
    os.makedirs(out_dir, exist_ok=True)
    ds, name, key = load_ds(mat_path)
    cfg = CONFIG[key]
    att = ds.attractor
    print(f"[2D] {ds.K} Gaussians, attractor {np.round(att, 3)}")
    print("[2D] max eig of sym(A_k):", ["%.1e" % v for v in ds.check_stability()], "(all must be < 0)")

    x_range, y_range = cfg["x_range"], cfg["y_range"]
    starts = np.array(cfg["starts"], dtype=float).T

    # (a) nominal DS: streamlines + trajectories, no obstacle
    fig, ax = plt.subplots(figsize=(7, 5))
    X, Y = np.meshgrid(np.linspace(*x_range, 60), np.linspace(*y_range, 45))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j], V[i, j] = ds(np.array([X[i, j], Y[i, j]]))
    ax.streamplot(X, Y, U, V, color="0.6", density=1.2)
    for s in starts.T:
        x = s.copy(); path = [x.copy()]
        for _ in range(3000):
            x = x + 0.005 * ds(x); path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], lw=2.5)
        ax.plot(*s, "k*", ms=10)
    ax.plot(*att, "r*", ms=18, label="attractor")
    ax.set(xlim=x_range, ylim=y_range, aspect="equal", title=f"Nominal {name}")
    ax.legend(loc="upper right")
    fig.savefig(f"{out_dir}/2d_nominal_ds.png", dpi=150, bbox_inches="tight")

    # (b) same DS modulated around an ellipse with the Lab 2 package.
    # Trajectories are integrated here with a small step (the package's own
    # streamline helper uses dt = 0.05 at constant speed 6, too coarse for a ~1 m workspace).
    obs = [Obstacle(a=list(cfg["obstacle_axes"]), p=[1, 1], x0=list(cfg["obstacle_center"]),
                    th_r=cfg["th_r"], sf=cfg["sf"])]
    fig, ax = Simulation_vectorFields(x_range, y_range, 50, obs, xAttractor=att,
                                      dynamicalSystem=ds, noTicks=False, figureSize=(7, 5))
    o = obs[0]
    R = np.array([[np.cos(o.th_r), -np.sin(o.th_r)], [np.sin(o.th_r), np.cos(o.th_r)]])
    print("[2D] start            min Gamma   final error")
    for s in starts.T:
        x = s.copy(); gmin = np.inf; path = [x.copy()]
        for _ in range(20000):
            x = obs_avoidance_rk4(0.002, x, obs, obs_avoidance=obs_avoidance_interpolation_moving,
                                  ds=ds, x0=att)
            path.append(x.copy())
            xt = R.T @ (x - np.array(o.x0))
            gmin = min(gmin, np.sum((xt / (o.sf * np.array(o.a))) ** 2))
            if np.linalg.norm(x - att) < 0.01:
                break
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], lw=3)
        ax.plot(*s, "k*", ms=13)
        print(f"[2D] {np.round(s, 2)!s:18} {gmin:9.3f}   {np.linalg.norm(x - att):.4f}")
    ax.set_title(f"Modulated {name} (Lab 2 package, patched)")
    fig.savefig(f"{out_dir}/2d_modulated_ds.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show(block=True)
    plt.close("all")


if __name__ == "__main__":
    run(show=True)
