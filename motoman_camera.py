import matplotlib.pyplot as plt

from init_scene import *

robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_shelf1.json'

physics, viewer = init(robot_xml, assets_dir, scene_json)
world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)

dt = 0.001
world.opt.timestep = dt

ee_pose = tf.euler_matrix(-np.pi / 2, 0, -np.pi / 2)
ee_pose[:3, 3] = world.body("btarget").pos

ik_solver = TracIKSolver(
    "./motoman/motoman_dual.urdf",
    "base_link",
    "motoman_left_ee",
)
t0 = time.time()
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
t1 = time.time()
print("ik-test:", t1 - t0, qout)

lctrl = get_ctrl_indices(world, ["sda10f/" + j for j in ik_solver.joint_names])

print(qinds)
while viewer.is_alive:
    data.ctrl[lctrl] = qout
    # mujoco.mj_step(world, data)
    physics.step()
    viewer.render()
viewer.close()

t0 = time.time()
img1 = physics.render(camera_id=0, depth=True)
img2 = physics.render(camera_id=0, segmentation=True)
print(time.time() - t0)
plt.imshow(np.log10(img1))
plt.show()
plt.imshow(img2[:, :, 0])
plt.show()
plt.imshow(img2[:, :, 1])
plt.show()
print(img2[:,:,0])
print(img2[:,:,1])
