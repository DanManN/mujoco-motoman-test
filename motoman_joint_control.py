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

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

world, data, viewer = init(robot_xml, assets_dir, scene_json)
dt = 0.001
world.opt.timestep = dt
ctrlr = joint_controller(world, data)

angle1 = 0
angle2 = 0
sign1 = 1
sign2 = 1
step = 0.5
while viewer.is_alive:
    u = ctrlr.generate(
        q=data.qpos[-15:],
        dq=data.qvel[-15:],
        target=[0, angle1, angle2, 0, 0, 0, 0, 0, 0, 0, angle1, angle2, 0, 0, 0],
    )

    data.ctrl[:-1] = u[:]
    mujoco.mj_step(world, data)
    viewer.render()

    angle1 += sign1 * step * dt
    if angle1 > 1 or angle1 < 0:
        sign1 *= -1
    angle2 += sign2 * step * dt
    if angle2 > 1 or angle2 < 0:
        sign2 *= -1

viewer.close()
