"""
Part 2 (workflow step 4): 3D point-mass simulation of the modulated DS.

For each scene: >= 5 trajectories (above / below / sides of the obstacle),
nominal vs modulated, Lyapunov V(t), Gamma(t), distance(t), a results table
and an empirical V_dot < 0 check of the nominal DS.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from ds3d import make_field, simulate, metrics, check_vdot_on_grid
from ds3d.scenes import SCENES
from ds3d.plotting import plot_scene_3d, plot_lyapunov_and_distance, plot_modulation_field


def run_scene(key, out_dir="figures", rho=1.0, dt=0.005, t_max=15.0, tol=1e-3, verbose=True, keep_open=False):
    scene = SCENES[key]()
    ds, obstacles = scene["ds"], scene["obstacles"]
    att = ds.attractor
    field_mod = make_field(ds, obstacles, rho=rho)
    results, nominal = {}, {}
    for label, x0 in scene["starts"].items():
        t, tr = simulate(field_mod, x0, att, dt=dt, t_max=t_max, tol=tol)
        results[label] = dict(t=t, traj=tr, metrics=metrics(t, tr, att, obstacles, tol))
        tn, trn = simulate(ds, x0, att, dt=dt, t_max=t_max, tol=tol)
        nominal[label] = dict(t=tn, traj=trn, metrics=metrics(tn, trn, att, obstacles, tol))

    lo = [l[0] for l in scene["lims"]]; hi = [l[1] for l in scene["lims"]]
    vdot_max = check_vdot_on_grid(ds, att, lo, hi, n=12)

    if verbose:
        print(f"\n=== {scene['name']} ===")
        print(f"max V_dot of nominal DS on a 12^3 grid: {vdot_max:.2e}  (< 0 => Lyapunov decrease)")
        print(f"{'start':16}{'min Gamma':>10}{'min dist [cm]':>15}{'nominal min dist':>18}"
              f"{'final err [mm]':>16}{'t_conv [s]':>11}{'V monotone':>11}")
        for label, r in results.items():
            m, mn = r["metrics"], nominal[label]["metrics"]
            print(f"{label:16}{m['min_gamma']:10.3f}{100*m['min_distance']:15.2f}{100*mn['min_distance']:18.2f}"
                  f"{1000*m['final_error']:16.2f}{m['t_converge']:11.2f}{str(m['V_monotonic']):>11}")

    os.makedirs(out_dir, exist_ok=True)
    k = scene["key"]
    show_nominal = key != "C"
    plot_scene_3d(scene, results, nominal if show_nominal else None, fname=f"{out_dir}/3d_scene{k}_trajectories.png")
    plot_lyapunov_and_distance(scene, results, nominal if show_nominal else None, fname=f"{out_dir}/3d_scene{k}_lyapunov_gamma.png")
    if key == "A":
        plot_modulation_field(scene, fname=f"{out_dir}/3d_scene{k}_field_slice.png")
    _write_table(scene, results, nominal, vdot_max, f"{out_dir}/3d_scene{k}_metrics.md")
    if not keep_open:          # keep the windows when --show is used
        plt.close("all")
    return scene, results, nominal


def _write_table(scene, results, nominal, vdot_max, path):
    lines = [f"### {scene['name']}", "",
             f"max V_dot (nominal DS, 12^3 grid) = {vdot_max:.2e}", "",
             "| start | min Γ | min distance [cm] | nominal min distance [cm] | final error [mm] | t_conv [s] | V monotone |",
             "|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        m, mn = r["metrics"], nominal[label]["metrics"]
        lines.append(f"| {label} | {m['min_gamma']:.3f} | {100*m['min_distance']:.2f} | {100*mn['min_distance']:.2f} | "
                     f"{1000*m['final_error']:.2f} | {m['t_converge']:.2f} | {m['V_monotonic']} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def run(out_dir="figures", scenes=("A", "B", "C"), show=False):
    for key in scenes:
        run_scene(key, out_dir, keep_open=show)
    if show:
        plt.show(block=True)


if __name__ == "__main__":
    run()
