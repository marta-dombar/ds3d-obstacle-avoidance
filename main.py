"""
main.py: reproduces every simulation and figure of the report.

    python main.py                 # everything, headless (figures/ + tables in the terminal)
    python main.py --part 2d       # Lab 1 DS + Lab 2 package, 2D (workflow steps 1-3)
    python main.py --part 3d       # 3D point-mass simulations, scenes A, B, C (step 4)
    python main.py --part robot    # Panda in PyBullet, scene A (steps 5-6)
    python main.py --part robot --gui          # same, with the PyBullet window in real time
    python main.py --part anim     # animated 3D trajectories for the video
    python main.py --show          # also open the matplotlib windows
"""
import argparse
import os
import sys

# Windows consoles default to cp1252; make printed output UTF-8 safe
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "2d", "3d", "robot", "anim"], default="all")
    ap.add_argument("--gui", action="store_true", help="PyBullet GUI (robot part)")
    ap.add_argument("--show", action="store_true", help="open matplotlib windows")
    ap.add_argument("--scene", default="A", help="scene for robot/anim parts (A, B or C)")
    ap.add_argument("--starts", nargs="*", default=None, help="robot: subset of start labels, e.g. above 'side (+x)'")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    if args.part in ("all", "2d"):
        from scripts.part1_2d import run as run2d
        run2d(show=args.show)
    if args.part in ("all", "3d"):
        from scripts.part2_3d import run as run3d
        run3d(show=args.show)
    if args.part in ("all", "robot"):
        from scripts.part3_robot import run as run_robot
        run_robot(args.scene, gui=args.gui, starts=args.starts)
    if args.part in ("all", "anim"):
        from scripts.animate_3d import run as run_anim
        run_anim(args.scene, show=args.show)
    print("\nFigures and tables written to ./figures")


if __name__ == "__main__":
    main()
