import re
import sys
import json
import glob
import time

import numpy as np

import mujoco
import mujoco_viewer
from dm_control import mjcf


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
        <!--<body name="cloud" pos="0 0 1.5">-->
          <!--<joint type="free"/>-->
          <geom name="p1" size=".01" type="sphere" rgba=".9 .1 .1 1" pos="0.00 0 1.5"/>
          <geom name="p2" size=".01" type="sphere" rgba=".9 .1 .1 1" pos="0.02 0 1.5"/>
          <geom name="p3" size=".01" type="sphere" rgba=".9 .1 .1 1" pos="0.04 0 1.5"/>
          <geom name="p4" size=".01" type="sphere" rgba=".9 .1 .1 1" pos="0.06 0 1.5"/>
          <geom name="p5" size=".01" type="sphere" rgba=".9 .1 .1 1" pos="0.08 0 1.5"/>
        <!--</body>-->
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
        )
    robot = mjcf.from_path(robot_xml)
    world_model.attach(robot)
    return world_model


def init_sim(world_model, ASSETS):
    fixed_xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world_model.to_xml_string())
    world = mujoco.MjModel.from_xml_string(fixed_xml_str, ASSETS)
    data = mujoco.MjData(world)
    return world, data


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


def set_joint_values_list(mdata, joint_vals):
    mdata.qpos = joint_vals


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
    ll = world.jnt_range[:, 0]
    ul = world.jnt_range[:, 1]

    angle = 0
    sign = 1
    while viewer.is_alive:
        # print(data.geom("p1"))
        # print(world.geom(1))
        # print(len(world.geom_bodyid))
        # print(world.geom_bodyid)
        # print(len(world.name_geomadr))
        # print(world.name_geomadr)
        # print(world.names)
        # data.geom("p1").xpos = [0, angle, 1.5]
        data.ctrl[0] = angle
        data.ctrl[1:] = 0
        # data.ctrl[:] = 0
        mujoco.mj_step(world, data)
        print(len(data.contact), colliding_body_pairs(data.contact, world))
        viewer.render()
        angle += sign * 0.0001
        if angle > 1.58:
            sign *= -1

    viewer.close()
