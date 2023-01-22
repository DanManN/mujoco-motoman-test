import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf
import transformations as tf

motoman_both_arms = [
    "sda10f/torso_joint_b1",
    "sda10f/arm_left_joint_1_s",
    "sda10f/arm_left_joint_2_l",
    "sda10f/arm_left_joint_3_e",
    "sda10f/arm_left_joint_4_u",
    "sda10f/arm_left_joint_5_r",
    "sda10f/arm_left_joint_6_b",
    "sda10f/arm_left_joint_7_t",
    "sda10f/arm_right_joint_1_s",
    "sda10f/arm_right_joint_2_l",
    "sda10f/arm_right_joint_3_e",
    "sda10f/arm_right_joint_4_u",
    "sda10f/arm_right_joint_5_r",
    "sda10f/arm_right_joint_6_b",
    "sda10f/arm_right_joint_7_t",
]


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
    base_pos = scene_dict['workspace']['pos']
    components = scene_dict['workspace']['components']
    world_model = mjcf.from_xml_string(
        """
    <mujoco model="World">
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="0.5 0.5" texuniform="true" specular="0" shininess="0" reflectance="0" emission="1" />
      </asset>
      <worldbody>
        <geom name="floor" size="2 2 .05" type="plane" material="grid" condim="3"/>
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" castshadow="false" diffuse="1 1 1"/>
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
        <body name="bpick" pos="0.8 0.3 1.05">
          <freejoint/>
          <geom name="gpick" size=".04 .04 .04" type="box" rgba=".5 .1 .5 1" mass="0.05" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )
    scene_body = world_model.worldbody.body['scene']
    # scene_body.pos = base_pos
    for component_name, component in components.items():
        shape = component['shape']
        shape = np.array(shape)
        component['pose']['pos'] = np.array(component['pose']['pos']) + np.array(base_pos)
        pos = np.array(component['pose']['pos'])
        ori = component['pose']['ori']  # x y z w
        scene_body.add(
            'geom',
            name=component_name,
            type='box',
            pos=pos,
            quat=ori,
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


def init_sim(world_model, ASSETS):
    fixed_xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world_model.to_xml_string())
    world = mujoco.MjModel.from_xml_string(fixed_xml_str, ASSETS)
    data = mujoco.MjData(world)
    return world, data


def get_qpos_indices(model, joints=motoman_both_arms):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints], dtype=int)
    return qpos_inds


def get_ctrl_indices(model, joints=motoman_both_arms):
    ctrl_inds = np.array([model.joint(j).id for j in joints], dtype=int)
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


def init(robot_xml, assets_dir, scene_json, gui=True):
    world_model = load_scene_workspace(robot_xml, scene_json)
    ASSETS = asset_dict(assets_dir)
    t0 = time.time()
    world, data = init_sim(world_model, ASSETS)
    t1 = time.time()
    print("init_sim:", t1 - t0)
    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None
    return world, data, viewer


if __name__ == '__main__':
    robot_xml = 'motoman/motoman.xml'
    assets_dir = 'motoman/meshes'
    scene_json = 'scene_shelf1.json'

    world, data, viewer = init(robot_xml, assets_dir, scene_json)
    qinds = get_qpos_indices(world)
    ictrl = get_ctrl_indices(world)

    dt = 0.001
    world.opt.timestep = dt

    ee_pose = tf.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
    ee_pose[:3, 3] = [0.72, 0.3, 1.05]

    print(world.joint(world.body("phys").jntadr[0]))
    print(world.joint(world.body("bpick").jntadr[0]))
    print(ictrl)

    angle1 = 1
    angle2 = 1
    while viewer.is_alive:
        mocap_id = world.body("btarget").mocapid
        data.mocap_pos[mocap_id] = [0.55, 0.55, 1 + angle1]
        # mujoco.mj_forward(world,data)
        target = [0, 0, angle2, 0, angle1, 0, 0, 0, 0, angle1, 0, angle2, 0, 0, 0]
        # u = ctrlr.generate(
        #     q=data.qpos[-15:],
        #     dq=data.qvel[-15:],
        #     target=target,
        # )
        data.ctrl[ictrl] = target

        mujoco.mj_step(world, data)

        viewer.render()
        if np.linalg.norm(data.qpos[qinds] - target) < 0.02:
            angle1 = 1 - angle1
            angle2 = 1 - angle2

    viewer.close()
