"""
Animated 3D view of the point-mass trajectories (for the demonstration video).
Trajectories grow over time while the camera slowly rotates.
Saved as MP4 if ffmpeg is available, otherwise as GIF.
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from ds3d import make_field, simulate
from ds3d.scenes import SCENES
from ds3d.plotting import draw_obstacles, set_equal, COLORS


def run(scene_key="A", out_dir="figures", n_frames=150, fps=30, show=False):
    scene = SCENES[scene_key]()
    att = scene["ds"].attractor
    field = make_field(scene["ds"], scene["obstacles"])
    trajs = {k: simulate(field, x0, att, dt=0.01, t_max=8)[1] for k, x0 in scene["starts"].items()}
    T = max(len(t) for t in trajs.values())

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_obstacles(ax, scene["obstacles"])
    ax.scatter(*att, marker="*", s=350, color="gold", edgecolor="k")
    set_equal(ax, scene["lims"])
    lines, heads = [], []
    for j, (lab, tr) in enumerate(trajs.items()):
        c = COLORS[j % len(COLORS)]
        ax.scatter(*tr[0], color=c, s=25, edgecolor="k")
        lines.append(ax.plot([], [], [], color=c, lw=2.5, label=lab)[0])
        heads.append(ax.plot([], [], [], "o", color=c, ms=7)[0])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(scene["name"] + "\nx_dot = M(x) f(x)")

    def update(i):
        k = int(min(T - 1, i / (n_frames - 20) * T))
        for line, head, tr in zip(lines, heads, trajs.values()):
            kk = min(k, len(tr) - 1)
            line.set_data(tr[:kk + 1, 0], tr[:kk + 1, 1]); line.set_3d_properties(tr[:kk + 1, 2])
            head.set_data([tr[kk, 0]], [tr[kk, 1]]); head.set_3d_properties([tr[kk, 2]])
        ax.view_init(elev=22, azim=-125 - 70 * i / n_frames)   # rotate towards a side view
        return lines + heads

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    os.makedirs(out_dir, exist_ok=True)
    if shutil.which("ffmpeg"):
        path = f"{out_dir}/anim_scene{scene_key}.mp4"
        ani.save(path, writer=FFMpegWriter(fps=fps), dpi=110)
    else:
        path = f"{out_dir}/anim_scene{scene_key}.gif"
        ani.save(path, writer=PillowWriter(fps=fps), dpi=80)
    print(f"[anim] saved {path}")
    if show:
        plt.show(block=True)
    plt.close(fig)
    return path


if __name__ == "__main__":
    run(show=True)
