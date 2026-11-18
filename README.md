Laboratory Project: 3D Obstacle Avoidance and Control using Dynamical Systems.
Robot's Control Architectures.
Universidad Carlos III de Madrid.

Team Group 8: Jaime Solé Pérez; Erika Santano Martin; Marta Domínguez Bartolomé.
______________________________________________________________________________________________________________


An autonomous dynamical system [Derivate (x) = f(x)] in R³ with a globally stable attractor [x*] is modulated around 3D obstacles, [Derivate (x) = M(x)·f(x)] with [M = E·D·E^(-1)], and used as the trajectory generator of the end effector of a Franka Emika Panda in PyBullet [Derivate (q) = J†(q)·Derivate(x)mod].



INSTALLATION:

Python 3.10 – 3.13. Use an isolated environment (conda or venv), as in the Lab 2 guide.

```bash
#conda
conda create -n ds3d python=3.11 -y
conda activate ds3d
pip install -r requirements.txt

#venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


- requirements.txt also installs the patched Lab 2 package (./exercise2_obstacle) in editable mode.



RUNNING:

bash
python main.py                      				# everything; results: figures/ and result tables
python main.py --part 2d            				# steps 1 to 3: Lab 1 DS in the Lab 2 package (2D)
python main.py --part 3d            				# step 4: 3D point mass simulations, scenes A, B, C
python main.py --part robot --gui   				# steps 5 to 6: Panda in PyBullet, real time, with window
python main.py --part robot --gui --starts above "side (+x)"   	# some starts
python main.py --part anim          				# rotating 3D animation
python main.py --part 3d --show     				# open interactive matplotlib windows




REPOSITORY STRUCTURE:


main.py                      reproduces all results
ds3d/
  dynamics.py                SpiralDS, LPVDS (Lab 1), LiftedDS (2D -> 3D), ScaledDS, Lyapunov checks
  obstacle.py                Ellipsoid / sphere: Gamma(x), grad Gamma, normal n(x), surface, distance
  modulation.py              Basis E = [n·e1·e2], D(Gamma), M = E·D·E^T, modulated velocity
  simulate.py                RK4 integration and metrics: min Gamma, distance, convergence, V(t)
  scenes.py                  Scenes A (spiral and sphere), B (lifted LPV + ellipsoid), C (saddle point)
  plotting.py                Report figures
robot/panda_pybullet.py      Panda controller: [Derivate (q) = J†(q)·Derivate(x)mod], joint limits, link distance logging
scripts/
  part1_2d.py                2D: Lab 1 DS plugged into the Lab 2 package
  part2_3d.py                3D point mass simulations, figures and tables
  part3_robot.py             Robot experiments, figures and tables
  animate_3d.py              Animation for the video
matlab/export_lpv_model.m    How to export the learned LPV-DS from Lab 1
exercise2_obstacle/          Lab 2 package (L. Huber, GPL-3.0) with some patches
figures/                     Generated results




KNOWN LIMITATIONS:

- Only the end effector point is modulated. The obstacle is inflated [sf = 2.2] to cover the gripper, which is enough when approaching from above or from the sides, but approaching from directly below makes the wrist touch the obstacle.
- A start exactly behind the obstacle on the line to the attractor is a saddle point of the modulated DS (C); any perturbation escapes it.
- [V = ½‖x-x*‖²] is a Lyapunov function of the nominal DS. With modulation it can temporarily increase (B), although all trajectories still converge.



REFERENCES:

Papers
1. S. M. Khansari-Zadeh, A. Billard, "A dynamical system approach to realtime obstacle avoidance", Autonomous Robots 32(4), pp. 433-454, 2012.
2. L. Huber, A. Billard, J.-J. Slotine, "Avoidance of convex and concave obstacles with convergence ensured through contraction", IEEE RA-L 4(2), pp. 1462-1469, 2019.
3. N. Figueroa, A. Billard, "Learning globally stable linear parameter varying dynamical systems", RSS 2018 (LPV-DS).
4. N. Figueroa, A. Billard, "A physically-consistent Bayesian non-parametric mixture model for dynamical system learning", CoRL 2018 (PC-GMM).
5. S. M. Khansari-Zadeh, A. Billard, "Learning stable nonlinear dynamical systems with Gaussian mixture models", IEEE T-RO 27(5), pp. 943-957, 2011 (SEDS).
Third-party code, libraries and models
6. Lab 1 code: EPFL-LASA icra19-lfd-tutorial-exercises (exercise1_learning: gui_lpvDS, ds-opt, phys-gmm), https://github.com/epfl-lasa/icra19-lfd-tutorial-exercises, with YALMIP and SeDuMi 1.3. Used to learn lpv_model.mat.
7. Lab 2 code: L. Huber, dynamic_obstacle_avoidance (exercise2_obstacle, GPL-3.0), included in exercise2_obstacle/ with small patches so it accepts a custom dynamical system.
8. E. Coumans, Y. Bai, PyBullet, https://pybullet.org; Franka Emika Panda model franka_panda/panda.urdf from pybullet_data.
9. NumPy, SciPy, Matplotlib.



GENERATIVE AI USE:

Parts of this code and documentation were written with the help of Claude (Anthropic, Claude Opus 5.5): code structure, debugging of the Lab 2 package, and test scripts. All results were generated, checked and run with the code in this repository.
