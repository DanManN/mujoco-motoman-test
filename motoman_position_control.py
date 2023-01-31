from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)
ictrl = get_ctrl_indices(world)

dt = 0.001
world.opt.timestep = dt

angle1 = 1
angle2 = 1
while viewer.is_alive:
    target = [0, angle1, angle2, 0, 0, 0, 0, 0, 0, 0, angle1, angle2, 0, 0, 0]
    data.ctrl[ictrl] = target
    # mujoco.mj_step(world, data)
    physics.step()
    viewer.render()

    if np.linalg.norm(data.qpos[qinds] - target) < 0.02:
        angle1 = 1 - angle1
        angle2 = 1 - angle2

viewer.close()
