import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf

from abr_control.controllers import OSC, Damping
from abr_control.arms.mujoco_config import MujocoConfig as arm

from init_scene import *


def cartesian_controller(world, data):
    robot_config = arm("motoman")

    def get_joint_pos_addrs(jntadr):
        # store the data.qpos indices associated with this joint
        first_pos = world.jnt_qposadr[jntadr]
        posvec_length = robot_config.JNT_POS_LENGTH[world.jnt_type[jntadr]]
        joint_pos_addr = list(range(first_pos, first_pos + posvec_length))[::-1]
        return joint_pos_addr

    def get_joint_dyn_addrs(jntadr):
        # store the data.qvel and .ctrl indices associated with this joint
        first_dyn = world.jnt_dofadr[jntadr]
        dynvec_length = robot_config.JNT_DYN_LENGTH[world.jnt_type[jntadr]]
        joint_dyn_addr = list(range(first_dyn, first_dyn + dynvec_length))[::-1]
        return joint_dyn_addr

    robot_config.N_JOINTS = world.nu
    robot_config.model = world
    robot_config.data = data
    robot_config._g = np.zeros(robot_config.N_JOINTS)
    robot_config._J3NP = np.zeros((3, world.nv))
    robot_config._J3NR = np.zeros((3, world.nv))
    robot_config._J6N = np.zeros((6, world.nu))
    robot_config._MNN = np.zeros((world.nv, world.nv))
    robot_config._R9 = np.zeros(9)
    robot_config._R = np.zeros((3, 3))
    robot_config._x = np.ones(4)
    robot_config.N_ALL_JOINTS = world.nv
    robot_config.joint_pos_addrs = []
    robot_config.joint_dyn_addrs = []
    for i in range(world.nu + 1):
        joint = world.jnt(i)
        name = joint.name
        if 'sda10f' in name:
            jntadr = joint.id
            robot_config.joint_pos_addrs += get_joint_pos_addrs(jntadr)
            robot_config.joint_dyn_addrs += get_joint_dyn_addrs(jntadr)

    damping = Damping(robot_config, kv=10)
    ctrlr = OSC(
        robot_config,
        kp=30,
        kv=20,
        ko=180,
        null_controllers=[damping],
        vmax=[20, 20],  # [m/s, rad/s]
        # [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, False, False, False],
    )

    mujoco.mj_forward(world, data)  # need to run ik
    return ctrlr


robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

world, data, viewer = init(robot_xml, assets_dir, scene_json)
dt = 0.001
world.opt.timestep = dt
ctrlr = cartesian_controller(world, data)

while viewer.is_alive:
    u = ctrlr.generate(
        q=data.qpos[-15:],
        dq=data.qvel[-15:],
        target=[0.6, 0.2, 1.0],
        ref_frame="sda10f/EE_left",
    )
    data.ctrl[:] = u[:]
    mujoco.mj_step(world, data)
    viewer.render()
viewer.close()
