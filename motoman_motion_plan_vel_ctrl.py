from init_scene import *

robot_xml = './motoman/motoman_vc.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

## Intialization
t0 = time.time()
physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)

dt = 0.001
world.opt.timestep = dt
ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
iqpos = get_qpos_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
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
ee_pose[:3, 3] = world.body("btarget").pos - [0.05, 0, 0]
print(ee_pose)
t0 = time.time()
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
ee_pose[:3, 3] += [0.05, 0, 0]
qout2 = ik_solver.ik(ee_pose, qinit=qout)
t1 = time.time()
print("ik:", t1 - t0, qout)

## Motion plan to joint goal
start = np.zeros(len(lctrl))
goal = qout
goal2 = qout2
t0 = time.time()
raw_plan = planner.plan(start, goal, 5, og.RRTConnect)
t1 = time.time()
print("motion plan:", t1 - t0, len(raw_plan))
raw_plan.append(goal2)

## velocity control
speed = 30
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
while viewer.is_alive:
    # data.ctrl[lctrl] = pos_traj[i]
    # data.act[actrl] = pos_traj[i]
    # data.ctrl[vctrl] = vel_traj[i]
    data.ctrl[lctrl] = jnt_traj(t)
    data.ctrl[vctrl] = jnt_traj(t, 1)
    mujoco.mj_step(world, data)
    t += dt
    if t > jnt_traj.duration:
        t = jnt_traj.duration
    else:
        print(np.linalg.norm(data.qpos[iqpos] - data.ctrl[lctrl]))
    if i < len(ts_sample) - 1:
        i += 1

    viewer.render()

viewer.close()
