"""
Scenarios shared by the point-mass simulation (part 2) and the robot (part 3).

All coordinates are in metres, in the robot base frame of a Franka Emika Panda
(base at the origin, z up), so the very same scene is reused on the robot.
"""
import os
import numpy as np

from .dynamics import SpiralDS, LPVDS, LiftedDS, ScaledDS
from .obstacle import Ellipsoid, rotation_xyz


def start_through(ds, point, radius, dt=0.002):
    """Start whose *nominal* trajectory passes through `point`.

    Integrates the DS backwards in time (x_dot = -f(x)) from `point` until the
    distance to the attractor equals `radius`. Choosing `point` inside the
    obstacle guarantees that, without modulation, the trajectory would collide.
    """
    x = np.asarray(point, dtype=float)
    while np.linalg.norm(x - ds.attractor) < radius:
        x = x - dt * ds(x)
    return x


def scene_spiral_sphere():
    """Scene A: analytic spiral DS + sphere. Every nominal path would hit the sphere."""
    attractor = np.array([0.42, 0.30, 0.35])
    ds = SpiralDS(attractor, k=1.0, w=0.3, kz=1.0)
    sphere = Ellipsoid.sphere(center=[0.42, 0.02, 0.38], radius=0.09, sf=1.3)
    # Points inside the sphere (offset 3 cm from the centre) the nominal DS would cross
    through = {"above": [0, 0, 1], "below": [0, 0, -1], "side (+x)": [1, 0, 0],
               "side (-x)": [-1, 0, 0], "above, +x": [0.7, 0, 0.7], "below, -x": [-0.7, 0, -0.7]}
    starts = {k: start_through(ds, sphere.c + 0.03 * np.array(v), radius=0.55)
              for k, v in through.items()}
    return dict(name="A: spiral DS + sphere", key="A", ds=ds, obstacles=[sphere],
                starts=starts, lims=([0.2, 0.75], [-0.3, 0.4], [0.15, 0.6]))


def load_lpv2d(mat_path="lpv_model.mat"):
    if os.path.exists(mat_path):
        return LPVDS.from_mat(mat_path), "learned LPV-DS"
    return LPVDS.synthetic(), "synthetic LPV-DS"


# Scene B settings, in the 2D DS coordinates (z_ds = 0 is the plane of the demonstrations).
# 'obstacle' and 'starts' are chosen on / around the path of the demonstrations.
SCENE_B = {
    "learned LPV-DS": dict(
        scale=0.30, obstacle=[-0.85, 0.33, 0.0], axes=[0.035, 0.05, 0.07], kz=1.0,
        starts={"side (left)": [-1.75, 0.75, 0.0], "side (right)": [-1.2, -0.05, 0.0],
                "middle": [-1.5, 0.45, 0.0]},
        # starts whose nominal path crosses the obstacle 2 cm above / below its centre
        through={"above": [0, 0, 0.02], "below": [0, 0, -0.02]}),
    "synthetic LPV-DS": dict(
        scale=0.15, obstacle=[-1.3, 0.8, 0.0], axes=[0.05, 0.07, 0.09], kz=1.0,
        starts={"side (left)": [-2.3, 1.2, 0.0], "side (right)": [-2.0, 0.2, 0.0],
                "above": [-2.1, 0.7, -1.3], "below": [-2.1, 0.7, 1.3], "far": [-2.3, -1.2, 0.5]}),
}


def scene_lpv_ellipsoid(mat_path="lpv_model.mat"):
    """Scene B: Lab 1 LPV-DS lifted to 3D, scaled into the workspace, + rotated ellipsoid.

    DS plane (x_ds, y_ds) -> robot (y, x); z_ds -> -z  (a proper rotation, det = +1),
    so 'above' starts use z_ds < 0.
    """
    lpv, label = load_lpv2d(mat_path)
    cfg = SCENE_B[label]
    scale = cfg["scale"]
    lifted = LiftedDS(lpv, z_star=0.0, kz=cfg["kz"])
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    target = np.array([0.50, 0.30, 0.35])
    offset = target - scale * R @ lifted.attractor
    ds = ScaledDS(lifted, scale=scale, offset=offset, R=R)

    def to_robot(p_ds):
        return scale * R @ np.asarray(p_ds, float) + offset

    obstacles = [Ellipsoid(center=to_robot(cfg["obstacle"]), axes=cfg["axes"],
                           R=rotation_xyz(0.3, 0.0, 0.5), sf=1.2)]
    starts = {k: to_robot(v) for k, v in cfg["starts"].items()}
    radius = np.linalg.norm(starts["middle"] - target) if "middle" in starts else 0.4
    for k, dv in cfg.get("through", {}).items():
        starts[k] = start_through(ds, obstacles[0].c + np.array(dv), radius)
    P = np.array(list(starts.values()) + [target, obstacles[0].c])
    lims = tuple([float(P[:, i].min() - 0.08), float(P[:, i].max() + 0.08)] for i in range(3))
    return dict(name=f"B: lifted {label} + ellipsoid", key="B", ds=ds, obstacles=obstacles,
                starts=starts, lims=lims)


def scene_saddle():
    """Scene C: start exactly behind the sphere on the line to the attractor.

    f(x) is parallel to the normal there, so the tangential part is zero and the
    trajectory stops on the surface (saddle point of the modulated DS). A 1 mm
    perturbation is enough to escape.
    """
    attractor = np.array([0.50, 0.30, 0.38])
    ds = SpiralDS(attractor, k=1.0, w=0.0, kz=1.0)        # purely linear DS
    obstacles = [Ellipsoid.sphere(center=[0.50, 0.02, 0.38], radius=0.09, sf=1.3)]
    starts = {"exactly behind": [0.50, -0.30, 0.38],
              "perturbed 1 mm": [0.501, -0.30, 0.38]}
    return dict(name="C: saddle point (linear DS)", key="C", ds=ds, obstacles=obstacles,
                starts=starts, lims=([0.3, 0.7], [-0.35, 0.35], [0.2, 0.56]))


SCENES = {"A": scene_spiral_sphere, "B": scene_lpv_ellipsoid, "C": scene_saddle}
