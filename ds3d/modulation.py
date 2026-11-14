"""
3D modulation of a dynamical system around obstacles.

    x_dot_mod = M(x) f(x),     M(x) = E(x) D(x) E(x)^-1

    E(x) = [ n(x)  e1(x)  e2(x) ]       orthonormal basis: normal + 2 tangents
    D(x) = diag(lambda_n, lambda_t, lambda_t)
    lambda_n = 1 - 1/Gamma^(1/rho)      -> 0 on the surface: no velocity into the obstacle
    lambda_t = 1 + 1/Gamma^(1/rho)      -> 2 on the surface: faster flow around it
    Far away (Gamma -> inf): D -> I, so M -> I and the nominal DS is recovered.

Because E is orthonormal, E^-1 = E^T (no matrix inversion needed).
"""
import numpy as np


def orthonormal_basis(n):
    """Return E = [n, e1, e2] (columns), a right-handed orthonormal basis of R^3.

    e1: Gram-Schmidt of a helper axis that is not parallel to n.
    e2: n x e1.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - (helper @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.column_stack([n, e1, e2])


def eigenvalues(gamma, rho=1.0):
    g = max(gamma, 1.0)               # numerical guard: never use Gamma < 1
    delta = 1.0 / g ** (1.0 / rho)
    return 1.0 - delta, 1.0 + delta


def modulation_matrix(x, f_x, obstacle, rho=1.0, tail_effect=False):
    """M(x) for one obstacle, plus the pieces used to build it (for logging).

    tail_effect=False (default): once the nominal velocity points away from the
    obstacle (n . f > 0) the normal component is NOT reduced (lambda_n = 1), so
    the robot does not slow down behind the obstacle.  Set True to always modulate.
    """
    gamma = obstacle.gamma(x)
    n = obstacle.normal(x)
    E = orthonormal_basis(n)
    lam_n, lam_t = eigenvalues(gamma, rho)
    if not tail_effect and n @ f_x > 0:
        lam_n = 1.0
    D = np.diag([lam_n, lam_t, lam_t])
    M = E @ D @ E.T
    return M, dict(gamma=gamma, n=n, E=E, D=D)


def modulated_velocity(x, ds, obstacles, rho=1.0, tail_effect=False, return_info=False):
    """x_dot_mod = M(x) f(x).

    Several obstacles: the modulation matrices are multiplied (Khansari-Zadeh &
    Billard 2012). This is exact for one obstacle and a common approximation for
    several obstacles that do not intersect.
    """
    x = np.asarray(x, dtype=float)
    f_x = ds(x)
    if not isinstance(obstacles, (list, tuple)):
        obstacles = [obstacles]
    M_total = np.eye(3)
    infos = []
    for obs in obstacles:
        M, info = modulation_matrix(x, f_x, obs, rho, tail_effect)
        M_total = M @ M_total
        infos.append(info)
    xd = M_total @ f_x
    if return_info:
        return xd, dict(M=M_total, f=f_x, obstacles=infos)
    return xd
