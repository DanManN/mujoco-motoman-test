#!/usr/bin/env python
""" convert collision meshes """
import sys
from math import pi
from glob import glob
import xml.etree.ElementTree as ET

import trimesh


def main(argv):
    for filename in glob(argv[1]):
        print(filename)
        mesh_obj = trimesh.load(filename, force='mesh')
        farr = filename.split('.')
        print(farr)
        newfile = '.'.join(farr[:-1] + ['stl'])
        print(newfile)
        with open(newfile, 'wb') as f:
            bits = trimesh.exchange.stl.export_stl(mesh_obj)
            # bits = trimesh.exchange.obj.export_obj(mesh_obj)
            # bits = trimesh.exchange.dae.export_collada(mesh_obj)
            f.write(bits)


if __name__ == '__main__':
    main(sys.argv)
