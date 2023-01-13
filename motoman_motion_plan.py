import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf

from abr_control.controllers import Joint
from abr_control.arms.mujoco_config import MujocoConfig as arm

from init_scene import *
from planner import Planner

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

world, data, viewer = init(robot_xml, assets_dir, scene_json)
dt = 0.001
world.opt.timestep = dt
ctrlr = joint_controller(world, data)


def collision_free(state):
    data.qpos[-15:] = [state[i] for i in range(15)]
    mujoco.mj_step1(world, data)
    return len(data.contact) <= 1


ll = world.jnt_range[1:, 0]
ul = world.jnt_range[1:, 1]
print(ll, ul)
planner = Planner(15, ll, ul, collision_free)
start = np.zeros(15)
goal = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  #np.random.uniform(ll, ul)
print(collision_free(start))
print(collision_free(goal))
plan = planner.plan(start, goal, 5)
# reset
data.qpos[-15:] = 0
mujoco.mj_forward(world, data)
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

    if np.linalg.norm(data.qpos[-15:] - target) < 0.1:
        i += 1

    viewer.render()

viewer.close()
