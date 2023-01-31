from init_scene import *

robot_xml = 'motoman/motoman.xml'
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
lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
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

## interpolate plan
speed = 0.1
steps = 1.0 / speed
plan = []
for x, y in zip(raw_plan[:-1], raw_plan[1:]):
    plan += np.linspace(x, y, int(np.linalg.norm(np.subtract(y, x)) * steps)).tolist()
print("interped plan:", len(plan))

## reset positions
data.qpos[qindl] = 0
mujoco.mj_forward(world, data)

## run simulation
i = 0
while viewer.is_alive:
    if i < len(plan):
        target = plan[i]
    else:
        if np.linalg.norm(data.qpos[qindl] - target) < speed:
            target = goal2

    data.ctrl[lctrl] = target
    # mujoco.mj_step(world, data)
    physics.step()

    if np.linalg.norm(data.qpos[qindl] - target) < speed:
        i += 1

    viewer.render()

viewer.close()
