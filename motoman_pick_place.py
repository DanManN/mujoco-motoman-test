import re
import sys
import json
import glob
import time

import numpy as np

from init_scene import *
from ompl import geometric as og

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_table1.json'

## Intialization
t0 = time.time()
world, data, viewer = init(robot_xml, assets_dir, scene_json)
# world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)

dt = 0.001
world.opt.timestep = dt
ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
qindl = get_qpos_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])

ll = world.jnt_range[lctrl, 0]
ul = world.jnt_range[lctrl, 1]


def collision_free(state):
    data.qpos[qindl] = [state[i] for i in range(ik_solver.number_of_joints)]
    mujoco.mj_step1(world, data)
    return len(data.contact) <= 1


planner = Planner(ik_solver.number_of_joints, ll, ul, collision_free)
t1 = time.time()
print("total init:", t1 - t0)


## IK for target position
def get_ik(pose, qinit, max_tries=100):
    c = 0
    while c < max_tries:
        q = ik_solver.ik(pose, qinit=qinit)
        if q is not None:
            break
        c += 1
    return q


data.qpos[get_objq_indices(world, "bpick")[:3]] *= [1, -1, 1]
mujoco.mj_step(world, data)

ee_pose = tf.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
ee_pose[:3, 3] = data.qpos[get_objq_indices(world, "bpick")[:3]
                           ] - [world.geom("gpick").size[0] + 0.05, 0, 0]
print(ee_pose)
t0 = time.time()
qout = get_ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
ee_pose[:3, 3] = data.qpos[get_objq_indices(world, "bpick")[:3]
                           ] - [world.geom("gpick").size[0], 0, 0]
qout2 = get_ik(ee_pose, qinit=qout)
ee_pose[:3, 3] += [0, 0, 0.1]
qout3 = get_ik(ee_pose, qinit=qout2)
ee_pose[:3, 3] = [0.75, -0.1, 1.1]
qout4 = get_ik(ee_pose, qinit=qout3)
ee_pose[:3, 3] -= [0.05, 0, 0]
qout5 = get_ik(ee_pose, qinit=qout4)
t1 = time.time()
print("ik:", t1 - t0, qout)

## Motion plan to joint goal
start = np.zeros(ik_solver.number_of_joints)
goal = qout
goal2 = qout2
goal3 = qout3
goal4 = qout4
goal5 = qout5
print(start, goal)
t0 = time.time()
raw_plan = planner.plan(start, goal, 10, og.RRTConnect)
raw_plan2 = planner.plan(goal3, goal4, 10, og.RRTConnect)
t1 = time.time()
print("motion plan:", t1 - t0, len(raw_plan))
if len(raw_plan) == 0:
    raw_plan = [start, goal]
if len(raw_plan2) == 0:
    raw_plan2 = [goal3, goal4]

speed = 0.1
## interpolate plan
steps = 1.0 / speed
plan = []
for x, y in zip(raw_plan[:-1], raw_plan[1:]):
    plan += np.linspace(x, y,
                        int(np.linalg.norm(np.subtract(y, x)) * steps)).tolist()
print("interped plan:", len(plan))
plan2 = []
for x, y in zip(raw_plan2[:-1], raw_plan2[1:]):
    plan2 += np.linspace(x, y, int(np.linalg.norm(np.subtract(y, x)) * steps
                                   )).tolist()
print("interped plan2:", len(plan2))

## reset positions
data.qpos[qindl] = 0
mujoco.mj_forward(world, data)
mocap_id = world.body("btarget").mocapid
data.mocap_pos[mocap_id] = ee_pose[:3, 3] - [0.05, 0, 0]

## run simulation
i = 0
suction = 0
while viewer.is_alive:
    if i < len(plan):
        target = plan[i]
    else:
        if np.linalg.norm(data.qpos[qindl] - goal) < 0.02:
            target = goal2
        if np.linalg.norm(data.qpos[qindl] - goal2) < 0.02:
            suction = 1
            target = goal3
        if np.linalg.norm(data.qpos[qindl] - goal3) < 0.02:
            i = 0
            plan = plan2
        if np.linalg.norm(data.qpos[qindl] - goal4) < 0.02:
            target = goal5
            suction = 0

    data.ctrl[lctrl] = target
    data.ctrl[0] = suction
    mujoco.mj_step(world, data)
    # physics.step()

    if np.linalg.norm(data.qpos[qindl] - target) < speed:
        i += 1

    viewer.render()

viewer.close()
