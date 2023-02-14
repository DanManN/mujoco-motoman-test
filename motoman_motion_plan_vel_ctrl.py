import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from init_scene import *

robot_xml = './motoman/motoman_vc.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

## Intialization
t0 = time.time()
physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data

viewer = mujoco_viewer.MujocoViewer(world, data)

qinds = get_qpos_indices(world)

dt = 0.001
world.opt.timestep = dt
ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
vctrl = get_ctrl_indices(world, ["sda10f/v_" + j for j in ik_solver.joint_names])
# vctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names], 'v_')
# actrl = get_act_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
qindl = get_qpos_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])

ll = world.jnt_range[lctrl, 0]
ul = world.jnt_range[lctrl, 1]


def collision_free(state):
    data.qpos[qindl] = [state[i] for i in range(len(qindl))]
    mujoco.mj_step1(world, data)
    return len(data.contact) <= 1


planner = Planner(len(qindl), ll, ul, collision_free)
t1 = time.time()
print("total init:", t1 - t0)

## IK for target position
ee_pose = tf.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
ee_pose[:3, 3] = world.body("btarget").pos
print(ee_pose)
t0 = time.time()
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
t1 = time.time()
print("ik:", t1 - t0, qout)

## Motion plan to joint goal
start = np.zeros(len(lctrl))
goal = qout
t0 = time.time()
raw_plan = planner.plan(start, goal, 5, og.RRTConnect)
t1 = time.time()
print("motion plan:", t1 - t0, len(raw_plan))

## velocity control
speed = 20
vel_limit = [-speed * np.pi / 180, speed * np.pi / 180]
acc_limit = [-500. * np.pi / 180, 500. * np.pi / 180]
import toppra as ta
import toppra.algorithm as ta_algo
import toppra.constraint as ta_constraint

ss = np.linspace(0, 1, len(raw_plan))
path = ta.SplineInterpolator(ss, raw_plan)
vlims = [vel_limit] * len(raw_plan[0])
alims = [acc_limit] * len(raw_plan[0])
pc_vel = ta_constraint.JointVelocityConstraint(vlims)
pc_acc = ta_constraint.JointAccelerationConstraint(alims)
instance = ta_algo.TOPPRA([pc_vel, pc_acc], path)
jnt_traj = instance.compute_trajectory()
ts_sample = np.linspace(0, jnt_traj.duration, int(jnt_traj.duration / dt))

print('duration: ', jnt_traj.duration)
qs_sample = jnt_traj(ts_sample)
qds_sample = jnt_traj(ts_sample, 1)
qdds_sample = jnt_traj(ts_sample, 2)
pos_traj = qs_sample.tolist()
vel_traj = qds_sample.tolist()
# pos_traj.append(jnt_traj(jnt_traj.duration))
# vel_traj.append(jnt_traj(jnt_traj.duration, 1))
# vel_traj.append([0] * len(vel_traj[0]))

## reset positions
data.qpos[qindl] = 0
mujoco.mj_forward(world, data)

## run simulation
i = 0
t = 0
planned_traj = []
real_traj = []
time_ax = []
while viewer.is_alive:
    # data.ctrl[lctrl] = pos_traj[i]
    # data.act[actrl] = pos_traj[i]
    # data.ctrl[vctrl] = vel_traj[i]
    q = jnt_traj(t)
    dq = jnt_traj(t, 1)
    data.ctrl[lctrl] = q
    data.ctrl[vctrl] = dq
    mujoco.mj_step(world, data)
    t += dt
    if t > jnt_traj.duration:
        t = jnt_traj.duration
    else:
        # print(np.linalg.norm(data.qpos[qindl] - data.ctrl[lctrl]))
        planned_traj.append(q)
        real_traj.append(data.qpos[qindl])
        time_ax.append(t - dt)
    if i < len(ts_sample) - 1:
        i += 1

    viewer.render()

viewer.close()

data = {}
for p, r, t in zip(planned_traj, real_traj, time_ax):
    for i in range(len(p)):
        data[f'plan_{i}'] = data.get(f'plan_{i}', []) + [p[i]]
        data[f'real_{i}'] = data.get(f'real_{i}', []) + [r[i]]
    data['time'] = data.get('time', []) + [t]

data = pd.DataFrame.from_dict(data)
data = pd.melt(data, ['time'])
data['type'] = [s[:-2] for s in data['variable']]
data['color'] = [s[-1] for s in data['variable']]
print(data)
ax = sb.lineplot(data, x='time', y='value', hue='color', style='type')
plt.show()
