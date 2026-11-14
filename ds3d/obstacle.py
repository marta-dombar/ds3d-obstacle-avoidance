"""
3D obstacles described by a distance function Gamma(x).

Ellipsoid with centre c, semi-axes a, orientation R and safety factor sf:

    x_t      = R^T (x - c)                       (point in obstacle frame)
    Gamma(x) = sum_i ( x_t,i / (sf a_i) )^(2p)   (p = 1 -> ellipsoid)

    Gamma < 1 : inside    Gamma = 1 : on the (inflated) surface    Gamma > 1 : outside

A sphere is the special case a = [r, r, r].
"""
import numpy as np


def rotation_xyz(roll=0.0, pitch=0.0, yaw=0.0):
    """Rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll) (angles in radians)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Ellipsoid:
    def __init__(self, center, axes, R=np.eye(3), sf=1.0, p=1):
        self.c = np.asarray(center, dtype=float)
        self.a = np.asarray(axes, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.sf = float(sf)
        self.p = int(p)

    @classmethod
    def sphere(cls, center, radius, sf=1.0):
        return cls(center, [radius] * 3, sf=sf)

    # ---- distance function and its gradient --------------------------------
    def to_local(self, x):
        return self.R.T @ (np.asarray(x, dtype=float) - self.c)

    def gamma(self, x):
        xt = self.to_local(x)
        return float(np.sum((xt / (self.sf * self.a)) ** (2 * self.p)))

    def grad_gamma(self, x):
        """Analytic gradient of Gamma, expressed in the world frame."""
        xt = self.to_local(x)
        A = self.sf * self.a
        g_local = 2 * self.p / A * (xt / A) ** (2 * self.p - 1)
        return self.R @ g_local

    def normal(self, x):
        """Unit outward normal n(x) = grad Gamma / ||grad Gamma||."""
        g = self.grad_gamma(x)
        nrm = np.linalg.norm(g)
        if nrm < 1e-12:  # only at the centre; any direction is fine
            return np.array([1.0, 0.0, 0.0])
        return g / nrm

    # ---- geometry helpers for plotting / metrics ---------------------------
    def surface(self, n_u=40, n_v=20, inflated=False):
        """Parametric surface (X, Y, Z) for plot_surface (p = 1 only)."""
        u = np.linspace(0, 2 * np.pi, n_u)
        v = np.linspace(0, np.pi, n_v)
        U, Vv = np.meshgrid(u, v)
        a = self.a * (self.sf if inflated else 1.0)
        P = np.stack([a[0] * np.cos(U) * np.sin(Vv),
                      a[1] * np.sin(U) * np.sin(Vv),
                      a[2] * np.cos(Vv)], axis=0).reshape(3, -1)
        P = self.R @ P + self.c[:, None]
        return [P[i].reshape(U.shape) for i in range(3)]

    def distance(self, x, n_u=120, n_v=60):
        """Euclidean distance from x to the *real* (not inflated) surface.

        Exact for a sphere; for an ellipsoid it is approximated by the closest
        point of a dense surface mesh (good to ~1 mm for the sizes used here).
        Negative values mean the point is inside.
        """
        x = np.asarray(x, dtype=float)
        if np.allclose(self.a, self.a[0]):
            return float(np.linalg.norm(x - self.c) - self.a[0])
        X, Y, Z = self.surface(n_u, n_v)
        d = np.sqrt((X - x[0]) ** 2 + (Y - x[1]) ** 2 + (Z - x[2]) ** 2).min()
        inside = np.sum((self.to_local(x) / self.a) ** 2) < 1
        return float(-d if inside else d)
