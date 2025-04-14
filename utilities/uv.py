import os
import trimesh
import torch
import numpy as np

def unwrap_uv(file_path):
    planar_uv = trimesh.visual.texture.planar_mapping(mesh)
    uv = np.array(planar_uv)
    return uv

def get_uvmap(file_path):
    mesh = trimesh.load_mesh(file_path)
    if not mesh.visual.kind == 'texture':
        print("Mesh does not have a texture!")
        exit()
    else:
        uv = np.array(mesh.visual.uv)
        #uv_torch = torch.from_numpy(uv)
    return uv