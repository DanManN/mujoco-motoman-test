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
    data = mujoco.MjData(world)
    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None
    t1 = time.time()
    print("init_sim:", t1 - t0)
    return world, data, viewer


if __name__ == '__main__':
    mesh_files = ['./meshes/airplane.obj', './meshes/mustard.obj']
    shutil.copy2(mesh_files[1], './meshes/TEMP.obj')
    world, data, viewer = init()
    renderer = mujoco.Renderer(world, 480, 640)

    dt = 0.001
    world.opt.timestep = dt

    gm = world.geom("mesh_geom")
    print(gm)
    mesh = world.mesh('dyn_mesh')
    print(mesh)

    step_num = 0
    while viewer.is_alive:
        mujoco.mj_step(world, data)
        step_num += 1
        if step_num % 1000 == 0:
            file_ind = step_num // 1000 % 2
            shutil.copy2(mesh_files[file_ind], './meshes/TEMP.obj')
            world = mujoco.MjModel.from_xml_path('./test.xml')
            # print(world.mesh('dyn_mesh'))
            mujoco.mjr_uploadMesh(world, renderer._mjr_context, mesh.id)

        glfw.make_context_current(viewer.window)
        viewer.render()
    viewer.close()
