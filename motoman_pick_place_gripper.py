from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_table1.json'

## Intialization
t0 = time.time()
world_model = load_scene_workspace(robot_xml, scene_json)
ASSETS = asset_dict(assets_dir)
to_attach = world_model.worldbody.body['bpick']
gm_attach = to_attach.geom['gpick']
attach_to = world_model.find('body', 'sda10f/EE_right')
attach_bd = attach_to.add(
    'body', name='attach_bd', pos=[0, -0.03, -0.03], quat=[1, 0, 0, 0]
)
attach_bd.add(
    'geom', name='attach_gm', type=gm_attach.type, size=gm_attach.size, group='3'
)
to_attach.remove()
t01 = time.time()
physics = init_sim(world_model, ASSETS)
world_a, data_a = physics.model._model, physics.data._data
t11 = time.time()
print(world_a.body('sda10f/attach_bd'))
print(world_a.geom('sda10f/attach_gm'))
print("init_sim_attached:", t11 - t01)
if False:
    viewer_a = mujoco_viewer.MujocoViewer(world_a, data_a)
    while viewer_a.is_alive:
        mujoco.mj_step(world_a, data_a)
        viewer_a.render()
    viewer_a.close()

physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)

dt = 0.001
world.opt.timestep = dt
ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_right_ee",
)
lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
qindl = get_qpos_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])
qindl_a = get_qpos_indices(world_a, ["sda10f/" + j for j in ik_solver.joint_names])

ll = world.jnt_range[lctrl, 0]
ul = world.jnt_range[lctrl, 1]


def collision_free(state):
    data.qpos[qindl] = [state[i] for i in range(ik_solver.number_of_joints)]
    mujoco.mj_step1(world, data)
    return len(data.contact) <= 1


def collision_attached(state):
    data_a.qpos[qindl_a] = [state[i] for i in range(ik_solver.number_of_joints)]
    mujoco.mj_step1(world_a, data_a)
    for pair in colliding_body_pairs(data_a.contact, world_a):
        if pair != ('sda10f/motoman_base', 'sda10f/torso_link_b1'):
            # print(pair)
            return False
    return True


planner = Planner(ik_solver.number_of_joints, ll, ul, collision_free)
planner_a = Planner(ik_solver.number_of_joints, ll, ul, collision_attached)
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


ee_pose = tf.euler_matrix(-np.pi / 2, -np.pi / 2, -np.pi / 2)
ee_pose[:3, 3] = world.body("bpick").pos - [world.geom("gpick").size[0] + 0.05, 0, 0]
print(ee_pose)
t0 = time.time()
qout = get_ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
ee_pose[:3, 3] = world.body("bpick").pos - [world.geom("gpick").size[0], 0, 0]
qout2 = get_ik(ee_pose, qinit=qout)
ee_pose[:3, 3] += [0, 0, 0.1]
qout3 = get_ik(ee_pose, qinit=qout2)
ee_pose[:3, 3] = [0.8, 0.1, 1.06]
qout4 = get_ik(ee_pose, qinit=qout3)
ee_pose[:3, 3] -= [0.06, 0, 0]
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
raw_plan = planner.plan(start, goal, 5, og.RRTConnect)
raw_plan2 = planner_a.plan(goal3, goal4, 5, og.RRTConnect)
t1 = time.time()
print("motion plan:", t1 - t0, len(raw_plan))

speed = 0.1
## interpolate plan
steps = 1.0 / speed
plan = []
for x, y in zip(raw_plan[:-1], raw_plan[1:]):
    plan += np.linspace(x, y, int(np.linalg.norm(np.subtract(y, x)) * steps)).tolist()
print("interped plan:", len(plan))
plan2 = []
for x, y in zip(raw_plan2[:-1], raw_plan2[1:]):
    plan2 += np.linspace(x, y, int(np.linalg.norm(np.subtract(y, x)) * steps)).tolist()
print("interped plan2:", len(plan2))

## reset positions
data.qpos[qindl] = 0
mujoco.mj_forward(world, data)
mocap_id = world.body("btarget").mocapid
data.mocap_pos[mocap_id] = ee_pose[:3, 3] - [0.05, 0, 0]

## run simulation
i = 0
grip = 0
while viewer.is_alive:
    if i < len(plan):
        target = plan[i]
    else:
        if np.linalg.norm(data.qpos[qindl] - goal) < 0.02:
            target = goal2
        if np.linalg.norm(data.qpos[qindl] - goal2) < 0.02:
            if grip < 255:
                grip += 1
            else:
                target = goal3
        if np.linalg.norm(data.qpos[qindl] - goal3) < 0.02:
            i = 0
            plan = plan2
        if np.linalg.norm(data.qpos[qindl] - goal4) < 0.02:
            target = goal5
            grip = 0

    data.ctrl[lctrl] = target
    data.ctrl[1] = grip
    # mujoco.mj_step(world, data)
    physics.step()
    # print(colliding_body_pairs(data.contact, world))

    if np.linalg.norm(data.qpos[qindl] - target) < speed:
        i += 1

    viewer.render()

viewer.close()
