"""
Point-mass simulation of the (modulated) DS and evaluation metrics.
"""
import numpy as np

from .modulation import modulated_velocity
from .dynamics import lyapunov


def make_field(ds, obstacles, rho=1.0, tail_effect=False, modulate=True):
    """Return the velocity field x -> x_dot used by the integrator."""
    if not modulate or not obstacles:
        return ds
    return lambda x: modulated_velocity(x, ds, obstacles, rho, tail_effect)


def rk4_step(field, x, dt):
    k1 = field(x)
    k2 = field(x + 0.5 * dt * k1)
    k3 = field(x + 0.5 * dt * k2)
    k4 = field(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(field, x0, attractor, dt=0.01, t_max=20.0, tol=1e-3):
    """Integrate x_dot = field(x) from x0 until ||x - x*|| < tol or t_max."""
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(int(t_max / dt)):
        x = rk4_step(field, x, dt)
        traj.append(x.copy())
        if np.linalg.norm(x - attractor) < tol:
            break
    traj = np.array(traj)
    t = np.arange(len(traj)) * dt
    return t, traj


def metrics(t, traj, attractor, obstacles, tol=1e-3):
    """Numbers for the results table of the report."""
    gam = np.array([[o.gamma(p) for o in obstacles] for p in traj]).min(axis=1)
    dist = np.array([[o.distance(p) for o in obstacles] for p in traj]).min(axis=1)
    err = np.linalg.norm(traj - attractor, axis=1)
    conv_idx = np.argmax(err < tol) if np.any(err < tol) else None
    V = lyapunov(traj, attractor)
    return dict(
        min_gamma=float(gam.min()),
        min_distance=float(dist.min()),
        final_error=float(err[-1]),
        converged=conv_idx is not None,
        t_converge=float(t[conv_idx]) if conv_idx is not None else np.nan,
        V_monotonic=bool(np.all(np.diff(V) <= 1e-12)),
        path_length=float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))),
        gamma=gam, distance=dist, V=V,
    )
