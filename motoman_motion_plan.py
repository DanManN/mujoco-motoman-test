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
from planner import Planner
from ompl import geometric as og

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

## Intialization
t0 = time.time()
world, data, viewer = init(robot_xml, assets_dir, scene_json)
dt = 0.001
world.opt.timestep = dt
ctrlr = joint_controller(world, data)
ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
ll = world.jnt_range[-15:, 0]
ul = world.jnt_range[-15:, 1]


def collision_free(state):
    data.qpos[-15:] = [state[i] for i in range(15)]
    mujoco.mj_step1(world, data)
    return len(data.contact) <= 1


planner = Planner(15, ll, ul, collision_free)
t1 = time.time()
print("total init:", t1 - t0)

## IK for target position
ee_pose = transformations.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
ee_pose[:3, 3] = [0.7, 0.3, 1.0]
t0 = time.time()
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
t1 = time.time()
print("ik:", t1 - t0, qout)

## Motion plan to joint goal
start = np.zeros(15)
goal = list(qout) + 7 * [0]
print(start, goal)
t0 = time.time()
raw_plan = planner.plan(start, goal, 5, og.RRTConnect)
t1 = time.time()
print("motion plan:", t1 - t0, len(raw_plan))

## interpolate plan
steps = 8
plan = []
for x, y in zip(raw_plan[:-1], raw_plan[1:]):
    plan += np.linspace(x, y, int(np.linalg.norm(np.subtract(y, x)) * steps)).tolist()
print("interped plan:", len(plan))

## reset positions
data.qpos[-15:] = 0
mujoco.mj_forward(world, data)

## run simulation
i = 0
while viewer.is_alive:
    if i < len(plan):
        target = plan[i]

    u = ctrlr.generate(
        q=data.qpos[-15:],
        dq=data.qvel[-15:],
        target=target,
    )
    data.ctrl[:] = u[:]
    mujoco.mj_step(world, data)

    if np.linalg.norm(data.qpos[-15:] - target) < 0.125:
        i += 1

    viewer.render()

viewer.close()
