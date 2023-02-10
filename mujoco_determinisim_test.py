from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_table1.json'

physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)
ictrl = get_ctrl_indices(world)

dt = 0.001
world.opt.timestep = dt

target = [
    0.89212142,
    1.19999647,
    -0.6510485,
    0.5845063,
    -2.34801669,
    -1.03796683,
    1.75305386,
    -1.65434289,
]

lctrl = get_ctrl_indices(world, motoman_right_arm)
d = 3
c = 10
while viewer.is_alive:
    data.ctrl[lctrl] = target
    # physics.step()
    mujoco.mj_step(world, data)
    viewer.render()
    c -= 1
    if c == 0:
        c = 30000
        d -= 1
        if d == 0:
            state2 = physics.get_state()
            break
        elif d == 1:
            state1 = physics.get_state()
            physics.reset()
            physics.set_state(init_state)
            # data.qacc_warmstart[:] = warmstart
        else:
            init_state = physics.get_state()
            # warmstart = data.qacc_warmstart
print(np.all(state1 == state2), np.linalg.norm(state1 - state2))
print(state1 - state2)
viewer.close()
