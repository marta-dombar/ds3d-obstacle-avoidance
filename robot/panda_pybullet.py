"""
Part 3: 3D modulated DS as trajectory generator for the
end-effector

Control loop:
    x          = forward kinematics of the end-effector 
    x_dot_mod  = M(x) f(x)                      
    x_dot_mod  = clip_norm(x_dot_mod, v_max)     
    w          = K_ori * orientation error       
    q_dot      = J^+(q) [x_dot_mod; w]          
    q_dot      = joint-velocity and joint-position limits
    send q_dot to the 7 arm joints (VELOCITY_CONTROL), step the simulator


"""
import time
import numpy as np
import pybullet as p
import pybullet_data

from ds3d.modulation import modulated_velocity

EE_LINK = 11                      
ARM = list(range(7))             
FINGERS = [9, 10]
MOVABLE = ARM + FINGERS           
DOWN = p.getQuaternionFromEuler([np.pi, 0, 0])  


class PandaDSController:
    def __init__(self, scene, gui=True, dt=1 / 240, v_max=0.25, damping=0.02,
                 k_ori=2.0, limit_margin=0.05, rho=1.0, real_time=True):
        self.scene, self.ds, self.obstacles = scene, scene["ds"], scene["obstacles"]
        self.dt, self.v_max, self.lam, self.k_ori = dt, v_max, damping, k_ori
        self.margin, self.rho, self.gui, self.real_time = limit_margin, rho, gui, real_time and gui

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        info = [p.getJointInfo(self.robot, j) for j in ARM]
        self.q_lo = np.array([i[8] for i in info])
        self.q_hi = np.array([i[9] for i in info])
        self.qd_max = np.array([i[11] for i in info])
        self._build_scene_visuals()
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.3, 60, -25, self.ds.attractor.tolist())

    # ------------------------------------------------------------------ 
    def _build_scene_visuals(self):
        self.obs_ids = []
        for o in self.obstacles:
            if np.allclose(o.a, o.a[0]):
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=o.a[0], rgbaColor=[0.69, 0.49, 0.49, 0.9])
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=o.a[0])
                orn = [0, 0, 0, 1]
            else:   # ellipsoid: unit sphere mesh scaled by the semi-axes
                verts, idx = _sphere_mesh()
                vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx, meshScale=o.a.tolist(),
                                          rgbaColor=[0.69, 0.49, 0.49, 0.9])
                col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, meshScale=o.a.tolist())
                orn = _rot_to_quat(o.R)
            
            self.obs_ids.append(p.createMultiBody(0, col, vis, o.c.tolist(), orn))
            p.setCollisionFilterGroupMask(self.obs_ids[-1], -1, 0, 0)   # robot passes through: we only measure
            # inflated (safety) boundary, drawn as a faint shell
            if self.gui:
                v2 = p.createVisualShape(p.GEOM_SPHERE, radius=o.sf * o.a.max(), rgbaColor=[0.3, 0.3, 0.3, 0.12])
                if np.allclose(o.a, o.a[0]):
                    p.createMultiBody(0, -1, v2, o.c.tolist())
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.84, 0, 1])
        p.createMultiBody(0, -1, vis, self.ds.attractor.tolist())

    # -------------------------------------------------------------
    def q_arm(self):
        return np.array([p.getJointState(self.robot, j)[0] for j in ARM])

    def ee_pose(self):
        s = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True)
        return np.array(s[4]), np.array(s[5])

    def jacobian(self):
        q_all = [p.getJointState(self.robot, j)[0] for j in MOVABLE]
        zeros = [0.0] * len(q_all)
        Jl, Ja = p.calculateJacobian(self.robot, EE_LINK, [0, 0, 0], q_all, zeros, zeros)
        return np.vstack([np.array(Jl)[:, :7], np.array(Ja)[:, :7]])     # 6 x 7

    def reset_to(self, x_start, n_iter=200):
        """Teleport the arm so that the end-effector is at x_start, gripper down (iterative IK)."""
        home = [0, -0.3, 0, -2.2, 0, 2.0, 0.79]
        for j, qj in zip(ARM, home):
            p.resetJointState(self.robot, j, qj)
        for j in FINGERS:
            p.resetJointState(self.robot, j, 0.02)
        for _ in range(n_iter):
            q = p.calculateInverseKinematics(self.robot, EE_LINK, list(x_start), DOWN,
                                             lowerLimits=self.q_lo.tolist() + [0, 0],
                                             upperLimits=self.q_hi.tolist() + [0.04, 0.04],
                                             jointRanges=(self.q_hi - self.q_lo).tolist() + [0.04, 0.04],
                                             restPoses=home + [0.02, 0.02], maxNumIterations=100)
            for j, qj in zip(ARM, q[:7]):
                p.resetJointState(self.robot, j, qj)
            if np.linalg.norm(self.ee_pose()[0] - x_start) < 1e-4:
                break
        p.setJointMotorControlArray(self.robot, FINGERS, p.POSITION_CONTROL, targetPositions=[0.02, 0.02])
        return np.linalg.norm(self.ee_pose()[0] - x_start)

    # ---------------------------------------------------------------- 
    def control(self, x, orn):
        xd = modulated_velocity(x, self.ds, self.obstacles, rho=self.rho)
        n = np.linalg.norm(xd)
        if n > self.v_max:
            xd = xd / n * self.v_max
        # orientation error (axis-angle) towards "gripper down"
        dq = p.getDifferenceQuaternion(orn.tolist(), DOWN)
        axis, ang = p.getAxisAngleFromQuaternion(dq)
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        w = self.k_ori * ang * np.array(axis)
        J = self.jacobian()
        if self.k_ori > 0:     
            v, J = np.concatenate([xd, w]), J
        else:                 
            v, J = xd, J[:3]
        qd = J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(len(v)), v)  
        s = np.max(np.abs(qd) / self.qd_max)
        if s > 1:
            qd /= s
       
        q = self.q_arm()
        q_next = q + qd * self.dt
        lo, hi = self.q_lo + self.margin, self.q_hi - self.margin
        qd[(q_next < lo) & (qd < 0)] = 0.0
        qd[(q_next > hi) & (qd > 0)] = 0.0
        return qd, xd

    def link_obstacle_distance(self):
        d = np.inf
        for oid in self.obs_ids:
            for cp in p.getClosestPoints(self.robot, oid, distance=1.0):
                d = min(d, cp[8])
        return d

    def run(self, x_start, t_max=15.0, tol=0.005, trail=True):
        err0 = self.reset_to(np.asarray(x_start, float))
        log = {k: [] for k in ["t", "x", "q", "qd", "xd", "gamma", "dist", "link_dist"]}
        prev = None
        for k in range(int(t_max / self.dt)):
            x, orn = self.ee_pose()
            qd, xd = self.control(x, orn)
            p.setJointMotorControlArray(self.robot, ARM, p.VELOCITY_CONTROL, targetVelocities=qd.tolist(),
                                        forces=[87] * 4 + [12] * 3)
            p.stepSimulation()
            if k % 4 == 0:
                log["t"].append(k * self.dt); log["x"].append(x); log["q"].append(self.q_arm())
                log["qd"].append(qd); log["xd"].append(xd)
                log["gamma"].append(min(o.gamma(x) for o in self.obstacles))
                log["dist"].append(min(o.distance(x) for o in self.obstacles))
                log["link_dist"].append(self.link_obstacle_distance())
                if self.gui and trail and prev is not None:
                    p.addUserDebugLine(prev.tolist(), x.tolist(), [0.1, 0.3, 0.9], 3, 0)
                prev = x
            if self.real_time:
                time.sleep(self.dt)
            if np.linalg.norm(x - self.ds.attractor) < tol:
                break
        log = {k: np.array(v) for k, v in log.items()}
        log["ik_start_error"] = err0
        log["q_lo"], log["q_hi"] = self.q_lo, self.q_hi
        return log

    def close(self):
        p.disconnect(self.cid)


def summarize(label, log, attractor):
    q = log["q"]
    margin = np.minimum(q - log["q_lo"], log["q_hi"] - q).min()
    err = np.linalg.norm(log["x"][-1] - attractor)
    return dict(start=label, min_gamma=log["gamma"].min(), min_dist_cm=100 * log["dist"].min(),
                min_link_dist_cm=100 * log["link_dist"].min(), final_err_mm=1000 * err,
                t_end=log["t"][-1], joint_limit_margin_rad=margin,
                max_qd_ratio=np.max(np.abs(log["qd"]) / np.array([2.175] * 4 + [2.61] * 3)))


# ---------------------------------------------------------------- 
def _sphere_mesh(n_u=24, n_v=12):
    verts, idx = [], []
    for i in range(n_v + 1):
        v = np.pi * i / n_v
        for j in range(n_u):
            u = 2 * np.pi * j / n_u
            verts.append([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    for i in range(n_v):
        for j in range(n_u):
            a, b = i * n_u + j, i * n_u + (j + 1) % n_u
            c, d = a + n_u, b + n_u
            idx += [a, c, b, b, c, d]
    return verts, idx


def _rot_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().tolist()
