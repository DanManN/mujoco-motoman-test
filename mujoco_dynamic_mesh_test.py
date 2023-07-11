import re
import sys
import json
import glob
import time
import shutil
import numpy as np

import glfw
import mujoco
import mujoco_viewer


def init(gui=True):
    t0 = time.time()
    world = mujoco.MjModel.from_xml_path('./mujoco_dynamic_mesh_test.xml')
    t1 = time.time()
    print("init_world:", t1 - t0)
    t0 = time.time()
    data = mujoco.MjData(world)
    t1 = time.time()
    print("init_data:", t1 - t0)
    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None
    return world, data, viewer


if __name__ == '__main__':
    mesh_files = ['./meshes/airplane.obj', './meshes/mustard.obj']
    shutil.copy2(mesh_files[1], './meshes/TEMP.obj')
    world, data, viewer = init()
    renderer = mujoco.Renderer(world, 480, 640)

    dt = 0.001
    world.opt.timestep = dt

    gm = world.geom("mesh_geom")
    # print(gm)
    mesh = world.mesh('dyn_mesh')
    # print(mesh)

    step_num = 0
    while viewer.is_alive:
        mujoco.mj_step(world, data)
        step_num += 1
        if step_num % 1000 == 0:
            file_ind = step_num // 1000 % 2
            t0 = time.time()
            shutil.copy2(mesh_files[file_ind], './meshes/TEMP.obj')
            t1 = time.time()
            print("I/O:", t1 - t0)
            t0 = time.time()
            world = mujoco.MjModel.from_xml_path(
                './mujoco_dynamic_mesh_test.xml'
            )
            t1 = time.time()
            print("update world:", t1 - t0)
            # print(world.mesh('dyn_mesh'))
            mujoco.mjr_uploadMesh(world, renderer._mjr_context, mesh.id)

        glfw.make_context_current(viewer.window)
        viewer.render()
    viewer.close()
