import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf

from tracikpy import TracIKSolver

from abr_control.utils import transformations
from abr_control.controllers import Joint
from abr_control.arms.mujoco_config import MujocoConfig as arm

from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

world, data, viewer = init(robot_xml, assets_dir, scene_json)
dt = 0.001
world.opt.timestep = dt
ctrlr = joint_controller(world, data)

ee_pose = transformations.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
ee_pose[:3, 3] = [0.6, 0.2, 1.0]

ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
t0 = time.time()
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
t1 = time.time()
print("ik-test:", t1 - t0, qout)

while viewer.is_alive:
    u = ctrlr.generate(
        q=data.qpos[-15:],
        dq=data.qvel[-15:],
        target=list(qout) + [0] * 7,
    )

    data.ctrl[:-1] = u[:]
    mujoco.mj_step(world, data)
    viewer.render()
viewer.close()
