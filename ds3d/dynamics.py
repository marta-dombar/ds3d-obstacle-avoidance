"""
Nominal (unmodulated) autonomous dynamical systems  x_dot = f(x).

Every DS here has a single attractor x* and a quadratic Lyapunov function
V(x) = 1/2 ||x - x*||^2.  Each DS can be written as  f(x) = A(x) (x - x*)  where
the symmetric part of A(x) is negative definite for every x, so

    V_dot = (x - x*)^T A(x) (x - x*) < 0   for all x != x*,

which proves global asymptotic stability (see the report, section 2.1).
"""
import numpy as np
from scipy.io import loadmat


# ----------------------------------------------------------------------------
# 1) Analytic 3D DS: linear "spiral" attractor
# ----------------------------------------------------------------------------
class SpiralDS:
    """f(x) = A (x - x*),  A = -[[k, -w, 0], [w, k, 0], [0, 0, kz]].

    A + A^T = -2 diag(k, k, kz) < 0  (if k, kz > 0), so V = 1/2||x-x*||^2 is a
    strict Lyapunov function whatever the rotation speed w is.
    """

    def __init__(self, attractor, k=1.0, w=1.5, kz=1.0):
        self.attractor = np.asarray(attractor, dtype=float)
        self.A = -np.array([[k, -w, 0.0], [w, k, 0.0], [0.0, 0.0, kz]])

    def __call__(self, x):
        return self.A @ (np.asarray(x, dtype=float) - self.attractor)


# ----------------------------------------------------------------------------
# 2) LPV-DS (Lab 1):  f(x) = sum_k gamma_k(x) (A_k x + b_k)
# ----------------------------------------------------------------------------
class LPVDS:
    """2D LPV-DS learned in Lab 1 (gui_lpvDS.m), or a synthetic one for testing.

    gamma_k(x) = pi_k N(x | mu_k, Sigma_k) / sum_j pi_j N(x | mu_j, Sigma_j)
    With the QLF constraint A_k + A_k^T < 0 and b_k = -A_k x*, every linear
    system shares the attractor x*, and V = 1/2||x - x*||^2 is a Lyapunov
    function of the mixture (convex combination of negative definite forms).
    """

    def __init__(self, priors, mu, sigma, A, b, attractor):
        self.priors = np.atleast_1d(np.asarray(priors, dtype=float))
        K = self.priors.size
        self.mu = np.asarray(mu, dtype=float).reshape(2, K)
        self.sigma = np.asarray(sigma, dtype=float).reshape(2, 2, K)
        self.A = np.asarray(A, dtype=float).reshape(2, 2, K)
        self.b = np.asarray(b, dtype=float).reshape(2, K)
        self.attractor = np.asarray(attractor, dtype=float).reshape(2)
        self.K = K
        # Precompute inverse covariances and normalisation constants
        self._sig_inv = np.stack([np.linalg.inv(self.sigma[:, :, k]) for k in range(K)], axis=2)
        self._norm = np.array([1.0 / np.sqrt((2 * np.pi) ** 2 * (abs(np.linalg.det(self.sigma[:, :, k]))
                                                                  + np.finfo(float).tiny))
                               for k in range(K)])

    # ---- loaders -----------------------------------------------------------
    @classmethod
    def from_mat(cls, path):
        """Load the file written by matlab/export_lpv_model.m."""
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        g = m["ds_gmm"]
        return cls(g.Priors, g.Mu, g.Sigma, m["A_k"], m["b_k"], m["att_g"])

    @classmethod
    def synthetic(cls):
        """A hand-made 2-component LPV-DS (stands in for your learned model).

        It produces an 'S'-shaped motion towards the origin. Each A_k satisfies
        A_k + A_k^T < 0 (checked in check_stability()).
        """
        att = np.zeros(2)
        mu = np.array([[-1.0, -0.4], [0.6, 0.2]]).T          # 2 x K
        sigma = np.stack([np.diag([0.25, 0.15]), np.diag([0.2, 0.2])], axis=2)
        A1 = np.array([[-1.0, 2.0], [-2.0, -1.0]])            # rotates clockwise
        A2 = np.array([[-1.0, -1.5], [1.5, -1.0]])            # rotates anticlockwise
        A = np.stack([A1, A2], axis=2)
        b = np.stack([-A1 @ att, -A2 @ att], axis=1)
        return cls([0.5, 0.5], mu, sigma, A, b, att)

    # ---- evaluation --------------------------------------------------------
    def gammas(self, x):
        """Posterior gamma_k(x), computed exactly as Lab 1 (posterior_probs_gmm.m):
        p_k = pi_k (N(x | mu_k, Sigma_k) + eps). The eps (2.2e-16) dominates far from
        the demonstrations, where the weights then tend to the priors pi_k."""
        d = x[:, None] - self.mu                               # 2 x K
        expo = np.einsum("ik,ijk,jk->k", d, self._sig_inv, d)
        pdf = self._norm * np.exp(-0.5 * expo)
        p = self.priors * (pdf + np.finfo(float).eps)
        return p / p.sum()

    def __call__(self, x, x0=None):
        # x0 is accepted (and ignored) so this matches the Lab 2 signature ds(x, x0)
        x = np.asarray(x, dtype=float)
        g = self.gammas(x)
        return np.einsum("k,ijk,j->i", g, self.A, x) + self.b @ g

    def check_stability(self):
        """Largest eigenvalue of (A_k + A_k^T)/2 for each k (must be < 0)."""
        return [np.max(np.linalg.eigvalsh(0.5 * (self.A[:, :, k] + self.A[:, :, k].T)))
                for k in range(self.K)]


class LiftedDS:
    """Lift a 2D DS to 3D:  f(x) = [ f2(x_xy) ; -kz (z - z*) ].

    V3 = V2(x, y) + 1/2 (z - z*)^2  =>  V3_dot = V2_dot - kz (z - z*)^2 < 0.
    """

    def __init__(self, ds2d, z_star=0.0, kz=1.0):
        self.ds2d = ds2d
        self.kz = kz
        self.attractor = np.array([*ds2d.attractor, z_star], dtype=float)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([*self.ds2d(x[:2]), -self.kz * (x[2] - self.attractor[2])])


class ScaledDS:
    """Map a DS from its own coordinates into the robot workspace.

    x_robot = s * R x_ds + o   =>   f_robot(x) = s R f(R^T (x - o) / s)
    A rotation + uniform scaling keeps V = 1/2||x - x*||^2 a Lyapunov function.
    """

    def __init__(self, ds, scale=1.0, offset=(0, 0, 0), R=np.eye(3)):
        self.ds, self.s, self.o, self.R = ds, float(scale), np.asarray(offset, float), np.asarray(R, float)
        self.attractor = self.s * self.R @ ds.attractor + self.o

    def __call__(self, x):
        x_ds = self.R.T @ (np.asarray(x, float) - self.o) / self.s
        return self.s * self.R @ self.ds(x_ds)


# ----------------------------------------------------------------------------
# Lyapunov helpers
# ----------------------------------------------------------------------------
def lyapunov(x, attractor):
    d = np.asarray(x) - attractor
    return 0.5 * np.sum(d * d, axis=-1)


def lyapunov_derivative(f, x, attractor):
    """V_dot(x) = grad V(x)^T f(x) = (x - x*)^T f(x)."""
    return float((np.asarray(x) - attractor) @ f(x))


def check_vdot_on_grid(f, attractor, lo, hi, n=15, exclude_radius=1e-3):
    """Empirical check: max of V_dot over a 3D grid (should be < 0)."""
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    worst = -np.inf
    for xx in axes[0]:
        for yy in axes[1]:
            for zz in axes[2]:
                p = np.array([xx, yy, zz])
                if np.linalg.norm(p - attractor) < exclude_radius:
                    continue
                worst = max(worst, lyapunov_derivative(f, p, attractor))
    return worst
