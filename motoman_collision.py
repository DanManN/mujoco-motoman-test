"""
construct a motoman scene, and test the collision methods in mujoco
"""
import re
import os
import sys
import json
import glob
import time
import numpy as np
import seaborn as sb
import transformations as tf
import matplotlib.pyplot as plt

import mujoco
import mujoco_viewer
from dm_control import mjcf

# scene_file = './motoman/scene.xml'
robot_xml = './motoman/mjmodel.xml'
scene_json = './scene_table1.json'

ASSETS = dict()
for fname in glob.glob('./motoman/meshes/*.stl'):
    with open(fname, 'rb') as f:
        ASSETS[fname] = f.read()


def load_scene_workspace():
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
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
      </asset>
      <worldbody>
        <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
        <light name="spotlight" mode="targetbodycom" target="scene" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 -20 4" cutoff="10"/>
        <body name="scene" pos= "0 0 0">
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
            rgba=[1., 0.64, 0.0, 1.0]
        )

    robot = mjcf.from_path(robot_xml)
    world_model.attach(robot)
    fixed_xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world_model.to_xml_string())
    return fixed_xml_str


def create_scene():
    world_xml = load_scene_workspace()
    world = mujoco.MjModel.from_xml_string(world_xml, ASSETS)
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


def test_collision_test():
    world, data = create_scene()

    # * set the robot configuration to a certain state
    # print out the mutable configurations
    position = data.qpos
    print('joint angle: ', position)

    joint_names = ['torso_joint_b1', 'arm_right_joint_1_s', 'arm_right_joint_2_l']
    print(get_joint_values(data, joint_names))
    joint_val_dict = {'torso_joint_b1': 0.1}
    set_joint_values(data, joint_val_dict)
    print('after setting joint values: ')
    print('joint angle: ', position)
    print(get_joint_values(data, joint_names))

    # * check collision
    t0 = time.time()
    mujoco.mj_step1(world, data)
    # mujoco.mj_collision(world, data)
    t1 = time.time()
    print(t1 - t0)
    print('collisions: ')
    print(len(data.contact), [(c.geom1, c.geom2) for c in data.contact])


def test_collision(
    joint_vals,
    mdata,
):
    """
    given a joint value list (within the joint range), check collision
    """
    # * set the robot configuration to a certain state
    # print out the mutable configurations
    set_joint_values_list(mdata, joint_vals)

    # * check collision by getting the output query of scene graph
    mujoco.mj_step1(world, mdata)
    # mujoco.mj_collision(world, mdata)
    # cols = [(c.geom1, c.geom2) for c in data.contact]
    return len(data.contact) > 1


display = len(sys.argv) > 1 and sys.argv[1][0] in ('t', 'T')
world, data = create_scene()
if display:
    viewer = mujoco_viewer.MujocoViewer(world, data)

# * generate a random joint angle within the range
world.jnt_range
ll = world.jnt_range[:, 0]
ul = world.jnt_range[:, 1]

num_samples = 1000
rand_joints = np.random.uniform(ll, ul, size=[num_samples] + list(ll.shape))

total_times = []
collisions = []
for i in range(num_samples):
    #     print('random joint: ', rand_joint)
    start_time_i = time.time()
    col = test_collision(rand_joints[i], data)
    duration_i = time.time() - start_time_i
    if display:
        viewer.render()
    total_times.append(duration_i)
    collisions.append(col)

print('done')
if display:
    viewer.close()
total_times = np.array(total_times)
collisions = np.array(collisions).astype(bool)
# * draw a statistics of the total time

plt.figure()
sb.boxplot(total_times)
plt.savefig('total_timing_boxplot.png')
print('collision timing')
plt.figure()
sb.boxplot(total_times[collisions])
plt.savefig('collision_timing_boxplot.png')
print('non-collision timing')
plt.figure()
sb.boxplot(total_times[collisions & 0])
plt.savefig('non_collision_timing_boxplot.png')
# number of collisions
print('number of collisions: ', collisions.astype(int).sum() / len(collisions))
