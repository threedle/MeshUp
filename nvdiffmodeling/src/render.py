# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from . import util
from . import mesh 
from . import renderutils as ru

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        light_pos,
        light_power,
        material,
        min_roughness
    ):

    ################################################################################
    # Texture lookups
    ################################################################################

    kd = material['kd'].sample(gb_texc, gb_texc_deriv)
    ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
    perturbed_nrm = None
    if 'normal' in material:
        perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3]

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material, "Material must specify a BSDF type"
    if material['bsdf'] == 'pbr':
        shaded_col = ru.pbr_bsdf(kd, ks, gb_pos, gb_normal, view_pos, light_pos, min_roughness) * light_power
    elif material['bsdf'] == 'diffuse':
        shaded_col = kd * ru.lambert(gb_normal, util.safe_normalize(light_pos - gb_pos)) * light_power
    elif material['bsdf'] == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif material['bsdf'] == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    else:
        assert False, "Invalid BSDF '%s'" % material['bsdf']

    out = torch.cat((shaded_col, alpha), dim=-1)
    return out

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        light_pos,
        light_power,
        resolution,
        min_roughness,
        spp,
        msaa
    ):

    full_res = resolution*spp

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, [resolution, resolution], mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, [resolution, resolution], mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texure coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################
    color = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, light_pos, light_power, mesh.material, min_roughness)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        color = util.scale_img_nhwc(color, [full_res, full_res], mag='nearest', min='nearest')

    # Return color & raster output for peeling
    return color


# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08,
        return_rast_map           = False,
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution*spp
    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    light_pos   = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)
    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), [resolution*spp, resolution*spp]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, light_pos, light_power, resolution, min_roughness, spp, msaa), rast)]
    if return_rast_map:
        return rast.detach()
    # Clear to background layer
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = util.scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accum_col = background
    else:
        accum_col = torch.zeros(size=(1, full_res, full_res, 3), dtype=torch.float32, device='cuda')

    # Composite BACK-TO-FRONT
    for color, rast in reversed(layers):
        alpha     = (rast[..., -1:] > 0) * color[..., 3:4]
        accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        if antialias:
            accum_col = dr.antialias(accum_col.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int()) # TODO: need to support bfloat16

    # Downscale to framebuffer resolution. Use avg pooling 
    out = util.avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col
    
    return out

def render_scene(
        ctx,
        meshes,  # List of meshes
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08,
        return_rast_map           = False,
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution * spp
    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    light_pos   = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos    = prepare_input_vector(view_pos)

    combined_v_pos = []
    combined_t_pos_idx = []
    combined_v_nrm = []
    combined_v_tng = []
    combined_v_tex = []
    combined_t_nrm_idx = []
    combined_t_tng_idx = []
    combined_t_tex_idx = []

    vertex_offset = 0
    uv_offset = 0  

    for mesh_ in meshes:
        v_pos = mesh_.v_pos[None, ...]
        combined_v_pos.append(v_pos)
        
        t_pos_idx_offset = mesh_.t_pos_idx + vertex_offset
        combined_t_pos_idx.append(t_pos_idx_offset)
        
        combined_v_nrm.append(mesh_.v_nrm[None, ...])
        combined_v_tng.append(mesh_.v_tng[None, ...])
        combined_v_tex.append(mesh_.v_tex[None, ...])

        combined_t_nrm_idx.append(mesh_.t_nrm_idx + vertex_offset)  # or alternative if face-based
        combined_t_tng_idx.append(mesh_.t_tng_idx + vertex_offset)  # adjust if needed

        if mesh_.t_tex_idx.shape == mesh_.v_tex.shape[:1]:  # If UV indices align with vertex count
            combined_t_tex_idx.append(mesh_.t_tex_idx + vertex_offset)
        else:
            combined_t_tex_idx.append(mesh_.t_tex_idx + uv_offset)

        # Update offsets
        vertex_offset += v_pos.shape[1]
        uv_offset += mesh_.v_tex.shape[1]  

    combined_v_pos = torch.cat(combined_v_pos, dim=1)
    combined_t_pos_idx = torch.cat(combined_t_pos_idx, dim=0)
    combined_v_nrm = torch.cat(combined_v_nrm, dim=1)
    combined_v_tng = torch.cat(combined_v_tng, dim=1)
    combined_v_tex = torch.cat(combined_v_tex, dim=1)
    combined_t_nrm_idx = torch.cat(combined_t_nrm_idx, dim=0)
    combined_t_tng_idx = torch.cat(combined_t_tng_idx, dim=0)
    combined_t_tex_idx = torch.cat(combined_t_tex_idx, dim=0)

    combined_mesh = mesh.Mesh(
        v_pos=combined_v_pos.squeeze(0),
        t_pos_idx=combined_t_pos_idx.squeeze(0),
        v_nrm=combined_v_nrm.squeeze(0),
        v_tng=combined_v_tng.squeeze(0),
        v_tex=combined_v_tex.squeeze(0),
        t_nrm_idx=combined_t_nrm_idx.squeeze(0),
        t_tng_idx=combined_t_tng_idx.squeeze(0),
        t_tex_idx=combined_t_tex_idx.squeeze(0),
        material=meshes[0].material  # Assuming all meshes share the same material, or pick a method to combine them
    )

    # Clip space transform
    v_pos_clip = ru.xfm_points(combined_mesh.v_pos.unsqueeze(0), mtx_in)

    # Render the entire scene at once
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, combined_mesh.t_pos_idx.int(), [full_res, full_res]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, combined_mesh, view_pos, light_pos, light_power, resolution, min_roughness, spp, msaa), rast)]

    if return_rast_map:
        return rast.detach()

    # Clear to background layer
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = util.scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accum_col = background
    else:
        accum_col = torch.zeros(size=(1, full_res, full_res, 3), dtype=torch.float32, device='cuda')

    # Composite BACK-TO-FRONT
    for color, rast in reversed(layers):
        alpha     = (rast[..., -1:] > 0) * color[..., 3:4]
        accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        if antialias:
            accum_col = dr.antialias(accum_col.contiguous(), rast, v_pos_clip, combined_mesh.t_pos_idx.int())

    # Downscale to framebuffer resolution. Use avg pooling 
    out = util.avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    return out

def render_scene_(
        ctx,
        meshes,  # List of meshes
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08,
        return_rast_map           = False,
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution * spp
    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    light_pos   = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos    = prepare_input_vector(view_pos)

    # Clear to background layer
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = util.scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accum_col = background
    else:
        accum_col = torch.zeros(size=(1, full_res, full_res, 3), dtype=torch.float32, device='cuda')

    # Render each mesh separately
    layers = []
        # Clip space transform for the current mesh
        
        # Render all layers front-to-back for the current mesh

    v_pos_clip_list = []
    for mesh_ in meshes:
        v_pos_clip = ru.xfm_points(mesh_.v_pos[None, ...], mtx_in)
        v_pos_clip_list.append(v_pos_clip)
    for _ in range(num_layers):
        for i, mesh_ in enumerate(meshes):
            with dr.DepthPeeler(ctx, v_pos_clip_list[i], mesh_.t_pos_idx.int(), [full_res, full_res]) as peeler:
                rast, db = peeler.rasterize_next_layer()
                color = render_layer(rast, db, mesh_, view_pos, light_pos, light_power, resolution, min_roughness, spp, msaa)
                layers += [(color, rast)]

    # Composite BACK-TO-FRONT across all meshes
    for color, rast in reversed(layers):
        alpha = (rast[..., -1:] > 0) * color[..., 3:4]
        accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        if antialias:
            accum_col = dr.antialias(accum_col.contiguous(), rast, v_pos_clip, mesh_.t_pos_idx.int())
        
    # Downscale to framebuffer resolution. Use avg pooling 
    out = util.avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    return out

