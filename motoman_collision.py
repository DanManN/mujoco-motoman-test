"""
construct a motoman scene, and test the collision methods in mujoco
"""
import seaborn as sb
import matplotlib.pyplot as plt

from init_scene import *


def test_collision(
    joint_inds,
    joint_vals,
    data,
):
    """
    given a joint value list (within the joint range), check collision
    """
    # * set the robot configuration to a certain state
    # print out the mutable configurations
    set_joint_values_list(data, joint_inds, joint_vals)

    # * check collision by getting the output query of scene graph
    mujoco.mj_step1(world, data)
    # mujoco.mj_collision(world, mdata)
    # cols = [(c.geom1, c.geom2) for c in data.contact]
    return len(data.contact) > 1


robot_xml = 'motoman/motoman.xml'
assets_dir = 'motoman/meshes'
scene_json = 'scene_table1.json'

gui = len(sys.argv) > 1 and sys.argv[1][0] in ('t', 'T')
world, data, viewer = init(robot_xml, assets_dir, scene_json, gui)
# world, data = physics.model._model, physics.data._data
qinds = get_qpos_indices(world)
ictrl = get_ctrl_indices(world)

# * generate a random joint angle within the range
world.jnt_range
ll = world.jnt_range[ictrl, 0]
ul = world.jnt_range[ictrl, 1]
print(ll)
print(ul)

num_samples = 1000
rand_joints = np.random.uniform(ll, ul, size=[num_samples] + list(ll.shape))

t0 = time.time()
total_times = []
collisions = []
for i in range(num_samples):
    #     print('random joint: ', rand_joint)
    start_time_i = time.time()
    col = test_collision(qinds, rand_joints[i], data)
    duration_i = time.time() - start_time_i
    if gui:
        viewer.render()
    total_times.append(duration_i)
    collisions.append(col)

print('done: ', time.time() - t0)
if gui:
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
