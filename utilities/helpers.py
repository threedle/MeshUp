"""
    Various helper functions

    create_scene -> combines multiple nvdiffmodeling meshes in to a single mesh with mega texture
"""
import sys
import numpy as np
import torch

from math import ceil

sys.path.append("../nvdiffmodeling")

import nvdiffmodeling.src.mesh as mesh
import nvdiffmodeling.src.texture as texture
import nvdiffmodeling.src.renderutils as ru
from NeuralJacobianFields import SourceMesh

cosine_sim = torch.nn.CosineSimilarity()

def cosine_sum(features, targets):
    return -cosine_sim(features, targets).sum()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()

    
def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a*scale_a)+add_a, a_idx
    elif a is None and b is not None:
        return (b*scale_b)+add_b, b_idx
    else:
        return torch.cat(((a*scale_a)+add_a, (b*scale_b)+add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

def create_scene(meshes, sz=1024):
    
    # Need to comment and fix code
    
    scene = mesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes)+1

    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz*ny)
    h = int(sz*nx)

    dev = meshes[0].v_pos.device

    kd_atlas = torch.ones ( (1, w, h, 4) ).to(dev)
    ks_atlas = torch.zeros( (1, w, h, 3) ).to(dev)
    kn_atlas = torch.ones ( (1, w, h, 3) ).to(dev)

    for i, m in enumerate(meshes):
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = int(i / ny)

        sc_x = 1./nx
        sc_y = 1./ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.,
            scale_b=torch.tensor([sc_x, sc_y]).to(dev),
            add_a=0.,
            add_b=torch.tensor([sc_x*pos_x, sc_y*pos_y]).to(dev)
        )

        kd_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['kd'].data.shape[-1]] = m.material['kd'].data
        ks_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['ks'].data.shape[-1]] = m.material['ks'].data
        kn_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['normal'].data.shape[-1]] = m.material['normal'].data

        scene = mesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene 
        )

    scene = mesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': texture.Texture2D(
                kd_atlas
            ),
            'ks': texture.Texture2D(
                ks_atlas
            ),
            'normal': texture.Texture2D(
                kn_atlas
            ),
        },
        base=scene # gets uvs etc from here
    )

    return scene

def get_vp_map_(v_pos, mtx_in, resolution):
    device = v_pos.device
    with torch.no_grad():
        vp_mtx = torch.tensor([
            [resolution / 2, 0., 0., (resolution - 1) / 2],
            [0., resolution / 2, 0., (resolution - 1) / 2],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.,]
        ], device=device)
        #4x4 extrinsic matrix @ world coordinates = num_verts,
        v_pos_clip = ru.xfm_points(v_pos[None, ...], mtx_in)
        v_pos_div = v_pos_clip / v_pos_clip[..., -1:]
        v_vp = (vp_mtx @ v_pos_div.transpose(1, 2)).transpose(1, 2)[..., :-1]
        # don't need manual z-buffer here since we're using the rast map to do occlusion
        if False:
            v_pix = v_vp[..., :-1].int().cpu().numpy()
            v_depth = v_vp[..., -1].cpu().numpy()
            # pix_v_map = -torch.ones(len(v_pix), resolution, resolution, dtype=int)
            pix_v_map = -np.ones((len(v_pix), resolution, resolution), dtype=int)
            # v_pix_map = resolution * torch.ones(len(v_pix), len(v_pos), 2, dtype=int)
            v_pix_map = resolution * np.ones_like(v_pix, dtype=int)
            # buffer = torch.ones_like(pix_v_map) / 0
            buffer = -np.ones_like(pix_v_map) / 0
            for i, vs in enumerate(v_pix):
                for j, (y, x) in enumerate(vs):
                    if x < 0 or x > resolution - 1 or y < 0 or y > resolution - 1:
                        continue
                    else:
                        if v_depth[i, j] > buffer[i, x, y]:
                            buffer[i, x, y] = v_depth[i, j]
                            if pix_v_map[i, x, y] != -1:
                                v_pix_map[i, pix_v_map[i, x, y]] = np.array([resolution, resolution])
                            pix_v_map[i, x, y] = j
                            v_pix_map[i, j] = np.array([x, y])
            v_pix_map = torch.tensor(v_pix_map, device=device)
        v_pix_map = v_vp[..., :-1].int()
        v_pix_map [(v_pix_map > resolution - 1) | (v_pix_map < 0)] = resolution
    return v_pix_map.long()



def get_vp_map(v_pos, mtx_in, resolution, fov):
    f = resolution / (2 * torch.tan(torch.deg2rad(fov/2)))
    device = v_pos.device
    with torch.no_grad():
        batch_size = mtx_in.shape[0]
        vp_mtx = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        vp_mtx[:,0,0] = f
        vp_mtx[:,1,1] = f
        vp_mtx[:,0,2] =  (resolution - 1) / 2
        vp_mtx[:,1,2] = (resolution - 1) / 2
        '''
        vp_mtx = torch.tensor([
            [f, 0., (resolution - 1) / 2],
            [0., f, (resolution - 1) / 2],
            [0., 0., 1.],
        ], device=device)
        '''
        #4x4 extrinsic matrix @ world coordinates = [num_verts, 4] 
        v_pos_clip = ru.xfm_points(v_pos[None, ...], mtx_in)
        v_pos_div = v_pos_clip[..., :-1] / v_pos_clip[..., -1:]
        v_vp = (vp_mtx @ v_pos_div.transpose(1, 2)).transpose(1, 2)[..., :-1]
        #breakpoint()
        # don't need manual z-buffer here since we're using the rast map to do occlusion
        if False:
            v_pix = v_vp[..., :-1].int().cpu().numpy()
            v_depth = v_vp[..., -1].cpu().numpy()
            # pix_v_map = -torch.ones(len(v_pix), resolution, resolution, dtype=int)
            pix_v_map = -np.ones((len(v_pix), resolution, resolution), dtype=int)
            # v_pix_map = resolution * torch.ones(len(v_pix), len(v_pos), 2, dtype=int)
            v_pix_map = resolution * np.ones_like(v_pix, dtype=int)
            # buffer = torch.ones_like(pix_v_map) / 0
            buffer = -np.ones_like(pix_v_map) / 0
            for i, vs in enumerate(v_pix):
                for j, (y, x) in enumerate(vs):
                    if x < 0 or x > resolution - 1 or y < 0 or y > resolution - 1:
                        continue
                    else:
                        if v_depth[i, j] > buffer[i, x, y]:
                            buffer[i, x, y] = v_depth[i, j]
                            if pix_v_map[i, x, y] != -1:
                                v_pix_map[i, pix_v_map[i, x, y]] = np.array([resolution, resolution])
                            pix_v_map[i, x, y] = j
                            v_pix_map[i, j] = np.array([x, y])
            v_pix_map = torch.tensor(v_pix_map, device=device)
        v_pix_map = v_vp.int().flip([-1])
        v_pix_map [(v_pix_map > resolution - 1) | (v_pix_map < 0)] = resolution
    return v_pix_map.long()

def occlude_vp_map(mesh, vp_map, rast_map, batch_size, res, device): 
    for idx, rast_faces in enumerate(rast_map[:, :, :, 3].view(batch_size, -1)):
        #get only the unique facces
        u_faces = rast_faces.unique().long()[1:] - 1
        t = torch.arange(len(mesh.v_pos), device=device)
        #faces with vertice indices[face_indices  we are looking for]=> yield all vertice points that are mapped from mesh
        u_ret = torch.cat([t, mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
        #u_ret[1] would be the overlappping counts bw all the vertices and the mapped vertices
        #no vertices are mapped to img if it the count in the list is just 1 
        non_verts = u_ret[0][u_ret[1] < 2]
        vp_map[idx][non_verts] = torch.tensor([res-1, res-1], device=device)
    return vp_map

def get_vertices_from_jacobians(source_mesh_path, jacobians, gpu=0):
    device = torch.device(f'cuda:{gpu}')

    source_mesh = SourceMesh.SourceMesh(0, source_mesh_path, {}, 1, ttype=torch.float)
    source_mesh.load()
    source_mesh.to(device)

    merged_jacobians_torch = torch.from_numpy(jacobians).to(device=device)
    vertices = source_mesh.vertices_from_jacobians(merged_jacobians_torch).squeeze()

    return vertices
