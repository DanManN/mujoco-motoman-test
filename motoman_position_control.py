import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf

from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

world, data, viewer = init(robot_xml, assets_dir, scene_json)
qinds = get_qpos_indices(world)
ictrl = get_ctrl_indices(world)

dt = 0.001
world.opt.timestep = dt

angle1 = 1
angle2 = 1
while viewer.is_alive:
    target = [0, angle1, angle2, 0, 0, 0, 0, 0, 0, 0, angle1, angle2, 0, 0, 0]
    data.ctrl[ictrl] = target
    mujoco.mj_step(world, data)
    viewer.render()

    if np.linalg.norm(data.qpos[qinds] - target) < 0.02:
        angle1 = 1 - angle1
        angle2 = 1 - angle2

viewer.close()
