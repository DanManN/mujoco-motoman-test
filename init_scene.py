import re
import sys
import json
import glob
import time

import numpy as np
import transformations as tf

import mujoco_viewer
from dm_control import mjcf
from dm_control import mujoco

from planner import Planner
from ompl import geometric as og
from tracikpy import TracIKSolver

motoman_right_arm = [
    "sda10f/arm_right_joint_1_s",
    "sda10f/arm_right_joint_2_l",
    "sda10f/arm_right_joint_3_e",
    "sda10f/arm_right_joint_4_u",
    "sda10f/arm_right_joint_5_r",
    "sda10f/arm_right_joint_6_b",
    "sda10f/arm_right_joint_7_t",
]

motoman_left_arm = [
    "sda10f/arm_left_joint_1_s",
    "sda10f/arm_left_joint_2_l",
    "sda10f/arm_left_joint_3_e",
    "sda10f/arm_left_joint_4_u",
    "sda10f/arm_left_joint_5_r",
    "sda10f/arm_left_joint_6_b",
    "sda10f/arm_left_joint_7_t",
]

motoman_both_arms = motoman_left_arm + motoman_right_arm

motoman_left_arm += ["sda10f/torso_joint_b1"]
motoman_right_arm += ["sda10f/torso_joint_b1"]
motoman_both_arms += ["sda10f/torso_joint_b1"]


def asset_dict(assets_dir):
    ASSETS = dict()
    for fname in glob.glob(assets_dir + '/*.stl'):
        with open(fname, 'rb') as f:
            ASSETS[fname] = f.read()
    return ASSETS


def load_scene_workspace(robot_xml, scene_json):
    scene_dict = None
    with open(scene_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Could not read file:", scene_json)
        return
    world_model = mjcf.from_xml_string(
        """
    <mujoco model="World">
      <option>
        <flag warmstart="disable" />
      </option>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="0.5 0.5" texuniform="true" specular="0" shininess="0" reflectance="0" emission="1" />
      </asset>
      <worldbody>
        <geom name="floor" size="2 2 .05" type="plane" material="grid" condim="3"/>
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" castshadow="false" diffuse="1 1 1"/>
        <body name="body_cam" pos="0.4 0 1.1" xyaxes="0 -1 0 0 0 1" mocap="true">
          <camera name="cam" fovy="90"/>
          <geom name="geom_cam" size="0.04 0.04 0.01" type="box" rgba="0 0 0 1" contype="2" conaffinity="2"/>
        </body>
        <body name="scene" pos="0 0 0">
        </body>
        <body name="phys" pos="-0.5 0 0">
          <freejoint/>
          <geom name="p1" size=".1" type="sphere" rgba=".9 .1 .1 1" pos="0.00 0 1.5"/>
        </body>
        <body name="obs" mocap="true" pos="0.55 0.55 1.0">
          <geom name="gobs" size="0.12" type="sphere" rgba=".9 .1 .1 1"/>
        </body>
        <body name="btarget" mocap="true" pos="0.7 0.3 1.0">
          <geom name="gtarget" size="0.02" type="sphere" rgba=".1 .9 .1 1" contype="2" conaffinity="2"/>
        </body>
        <body name="bpick" pos="0.8 -0.3 1.05">
          <freejoint/>
          <geom name="gpick" size=".03 .03 .03" type="box" rgba=".5 .1 .5 1" mass="0.05" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )
    scene_body = world_model.worldbody.body['scene']
    scene_body.pos = scene_dict['workspace']['pos']
    scene_body.quat = np.roll(scene_dict['workspace']['ori'], 1)

    components = scene_dict['workspace']['components']
    for component_name, component in components.items():
        shape = np.array(component['shape'])
        scene_body.add(
            'geom',
            name=component_name,
            type='box',
            pos=component['pose']['pos'],
            quat=np.roll(component['pose']['ori'], 1),
            size=shape / 2,
            rgba=[1., 0.64, 0.0, 1.0],
            # gap=10,
        )

    # size = 0.01
    # len_grid = 44
    # side = np.linspace(-2 * size * len_grid, 2 * size * len_grid, len_grid)
    # coords = [[x, y] for x in side for y in side]
    # for i in range(len_grid**2):
    #     world_model.worldbody.add(
    #         'body',
    #         name=f'cp{i}',
    #         pos=[*coords[i], -.3],
    #     ).add(
    #         'geom',
    #         type='sphere',
    #         condim=1,
    #         gap=size,
    #         size=[size],
    #         rgba=[.1, .1, .9, 1],
    #     )

    robot = mjcf.from_path(robot_xml)
    world_model.attach(robot)
    return world_model


def get_objq_indices(model, obj_name):
    jnt = model.joint(model.body(obj_name).jntadr[0])
    qpos_inds = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
    return qpos_inds


def get_qpos_indices(model, joints=motoman_both_arms):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds


def get_ctrl_indices(model, joints=motoman_both_arms):
    ctrl_inds = np.array([model.actuator(j.replace('joint_', '')).id for j in joints])
    return ctrl_inds


def get_joint_values(mdata, joint_names):
    # this assumes that all joints are revolute joints
    joint_val_dict = {}
    for name in joint_names:
        joint = mdata.joint(name)
        joint_val_dict[name] = joint.qpos[0]
    return joint_val_dict


def set_joint_values(mdata, joint_val_dict):
    # this assumes that all joints are revolute joints
    for name, val in joint_val_dict.items():
        joint = mdata.joint(name)
        joint.qpos[0] = val


def set_joint_values_list(mdata, joint_inds, joint_vals):
    mdata.qpos[joint_inds] = joint_vals


def colliding_body_pairs(contact, world):
    pairs = [
        (
            world.body(world.geom(c.geom1).bodyid[0]).name,
            world.body(world.geom(c.geom2).bodyid[0]).name
        ) for c in contact
    ]
    return pairs


def init_sim(world_model, ASSETS):
    fixed_xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world_model.to_xml_string())
    # world = mujoco.MjModel.from_xml_string(fixed_xml_str, ASSETS)
    # data = mujoco.MjData(world)
    physics = mujoco.Physics.from_xml_string(fixed_xml_str, ASSETS)
    return physics


def init(robot_xml, assets_dir, scene_json, gui=True):
    world_model = load_scene_workspace(robot_xml, scene_json)
    ASSETS = asset_dict(assets_dir)
    t0 = time.time()
    physics = init_sim(world_model, ASSETS)
    t1 = time.time()
    print("init_sim:", t1 - t0)
    viewer = mujoco_viewer.MujocoViewer(
        physics.model._model, physics.data._data
    ) if gui else None
    return physics, viewer


if __name__ == '__main__':
    robot_xml = 'motoman/motoman.xml'
    assets_dir = 'motoman/meshes'
    scene_json = 'scene_table1.json'

    physics, viewer = init(robot_xml, assets_dir, scene_json)
    world, data = physics.model._model, physics.data._data
    qinds = get_qpos_indices(world)
    ictrl = get_ctrl_indices(world)

    dt = 0.001
    world.opt.timestep = dt

    ee_pose = tf.euler_matrix(-np.pi / 2, -np.pi / 2, -np.pi / 2)
    ee_pose[:3, 3] = world.body("bpick").pos

    bd = world.body("phys")
    jnt = world.joint(bd.jntadr[0])

    ik_solver = TracIKSolver(
        "./motoman/motoman_dual.urdf",
        "base_link",
        "motoman_right_ee",
    )
    t0 = time.time()
    qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
    t1 = time.time()
    print("ik-test:", t1 - t0, qout)

    lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])

    # print(mujoco.MjModel.njnt)
    for i in range(world.njnt):
        print(i, world.jnt(i))
    for i in range(world.nu):
        print(i, world.actuator(i))
    # stype = world.geom(world.body(world.jnt(i).bodyid[0]).geomadr[0]).type[0]
    # print(stype, int(mujoco.mjtGeom.mjGEOM_BOX))

    z = 1.5
    sign = 1
    while viewer.is_alive:
        mocap_id = world.body("btarget").mocapid
        data.mocap_pos[mocap_id] = [0.55, 0.55, 1]
        data.qpos[get_objq_indices(world, "bpick")[:3]] = [0, 0, z]
        if z > 3 or z < 1.4:
            sign *= -1
        z += 0.01 * sign
        data.ctrl[1] = z*50
        data.ctrl[lctrl] = qout
        physics.step()
        # mujoco.mj_step(world, data)
        viewer.render()
    viewer.close()
