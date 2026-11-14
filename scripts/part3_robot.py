"""
Part 3 (workflow steps 5-7): Franka Panda in PyBullet driven by the modulated DS.

For the robot the obstacle is inflated (sf_robot) to cover the gripper envelope
(~10 cm): only the end-effector point is modulated, so the margin is what keeps
the hand away from the obstacle. The closest distance between *every* link and
the obstacle is logged to show where this single-point approach stops working.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from ds3d import make_field, simulate
from ds3d.scenes import SCENES
from ds3d.plotting import draw_obstacles, set_equal, COLORS
from robot.panda_pybullet import PandaDSController, summarize


def run(scene_key="A", gui=False, out_dir="figures", sf_robot=2.2, starts=None, t_max=12.0, k_ori=2.0):
    scene = SCENES[scene_key]()
    for o in scene["obstacles"]:
        o.sf = sf_robot
    att = scene["ds"].attractor
    labels = starts or list(scene["starts"].keys())

    ctrl = PandaDSController(scene, gui=gui, k_ori=k_ori)
    logs, rows = {}, []
    for lab in labels:
        if gui:
            print(f"[robot] start '{lab}'")
        log = ctrl.run(scene["starts"][lab], t_max=t_max)
        logs[lab] = log
        rows.append(summarize(lab, log, att))
    ctrl.close()

    # point-mass reference with the same (inflated) obstacle
    field = make_field(scene["ds"], scene["obstacles"])
    pm = {lab: simulate(field, scene["starts"][lab], att, dt=0.005, t_max=t_max)[1] for lab in labels}

    print(f"\n=== Robot: {scene['name']} (obstacle inflated sf = {sf_robot}) ===")
    hdr = f"{'start':12}{'EE min Gam':>11}{'EE dist[cm]':>12}{'link dist[cm]':>14}{'err[mm]':>9}{'t[s]':>6}{'q margin[rad]':>14}{'max|qd|/lim':>12}"
    print(hdr)
    for r in rows:
        flag = "  <-- link contact" if r["min_link_dist_cm"] < 0 else ""
        print(f"{r['start']:12}{r['min_gamma']:11.3f}{r['min_dist_cm']:12.1f}{r['min_link_dist_cm']:14.1f}"
              f"{r['final_err_mm']:9.1f}{r['t_end']:6.1f}{r['joint_limit_margin_rad']:14.2f}{r['max_qd_ratio']:12.2f}{flag}")

    os.makedirs(out_dir, exist_ok=True)
    _plots(scene, logs, pm, out_dir, scene_key)
    _table(scene, rows, sf_robot, f"{out_dir}/robot_scene{scene_key}_metrics.md")
    return logs, rows


def _plots(scene, logs, pm, out_dir, key):
    # 1) end-effector path vs point-mass path
    fig = plt.figure(figsize=(14, 6))
    for i, (title, el, az) in enumerate([("isometric", 22, -125), ("side (y-z)", 0, 0)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        draw_obstacles(ax, scene["obstacles"])
        for j, (lab, log) in enumerate(logs.items()):
            c = COLORS[j % len(COLORS)]
            ax.plot(*log["x"].T, color=c, lw=2.2, label=f"{lab} (robot EE)")
            ax.plot(*pm[lab].T, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.scatter(*scene["ds"].attractor, marker="*", s=300, color="gold", edgecolor="k")
        set_equal(ax, scene["lims"]); ax.view_init(el, az); ax.set_title(title)
        if el == 0:
            ax.set_xticks([]); ax.set_xlabel("")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Panda end-effector (solid) vs point-mass DS (dashed); wireframe = inflated safety envelope")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{out_dir}/robot_scene{key}_ee_paths.png", dpi=150, bbox_inches="tight")

    # 2) joints normalised to their limits, distances
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    lab0 = list(logs)[0]
    log = logs[lab0]
    qn = (log["q"] - log["q_lo"]) / (log["q_hi"] - log["q_lo"])
    for j in range(7):
        axs[0].plot(log["t"], qn[:, j], label=f"q{j+1}")
    axs[0].axhline(0, color="r", ls="--"); axs[0].axhline(1, color="r", ls="--")
    axs[0].set(title=f"joint positions / limits  (start '{lab0}')", xlabel="t [s]", ylabel="0 = lower, 1 = upper", ylim=(-0.05, 1.05))
    axs[0].legend(ncol=4, fontsize=7)
    for j, (lab, lg) in enumerate(logs.items()):
        c = COLORS[j % len(COLORS)]
        axs[1].plot(lg["t"], 100 * lg["dist"], color=c, label=lab)
        axs[2].plot(lg["t"], 100 * lg["link_dist"], color=c, label=lab)
    axs[1].axhline(0, color="r", ls="--")
    axs[1].set(title="end-effector distance to obstacle", xlabel="t [s]", ylabel="[cm]")
    axs[2].axhline(0, color="r", ls="--")
    axs[2].set(title="closest distance of ANY robot link to obstacle", xlabel="t [s]", ylabel="[cm]")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/robot_scene{key}_joints_distances.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def _table(scene, rows, sf, path):
    lines = [f"### Robot (Panda, PyBullet): {scene['name']}, obstacle inflated sf = {sf}", "",
             "| start | EE min Γ | EE min dist [cm] | min link dist [cm] | final error [mm] | time [s] | joint-limit margin [rad] | max abs(qd)/limit |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['start']} | {r['min_gamma']:.3f} | {r['min_dist_cm']:.1f} | {r['min_link_dist_cm']:.1f} | "
                     f"{r['final_err_mm']:.1f} | {r['t_end']:.1f} | {r['joint_limit_margin_rad']:.2f} | {r['max_qd_ratio']:.2f} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run(gui=True)
