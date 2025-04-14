import kornia
import os
import pathlib
import pymeshlab
import shutil
import torch
import torch.nn.functional as F
import torchvision
import logging
import yaml
import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

from easydict import EasyDict

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src import obj, util, mesh, render, texture, regularizer

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from utilities.uv import get_uvmap
except ModuleNotFoundError:
    print("No module named 'utilities.uv'")

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map, get_vp_map_, occlude_vp_map
from utilities.camera import CameraBatch, get_camera_params
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
from deepfloyd import DeepFloydGuidance
from utilities.io import load_ply

################################################################################

class Visualizer:
    def __init__(self, out_path: pathlib.Path, cfg: EasyDict, tb: SummaryWriter):
        self.out_path = out_path
        self.cfg = cfg
        self.tb = tb
        for d in ["figure", "images"]:
            os.makedirs(self.out_path / d, exist_ok=True)

    @torch.no_grad()
    def log_mesh(self, step: int, m: mesh.Mesh, cam_params: dict, video: Video, glctx, device):
        m = mesh.unit_size(m.eval(cam_params))
        img = render.render_mesh(
            glctx,
            m,
            cam_params["mvp"],
            cam_params["campos"],
            cam_params["lightpos"],
            self.cfg.log_light_power,
            self.cfg.log_res,
            1,
            background=torch.ones(1, self.cfg.log_res, self.cfg.log_res, 3, device=device),
        )
        video.ready_image(img)
        self.tb.add_mesh("predicted_mesh", vertices=m.v_pos.unsqueeze(0), faces=m.t_pos_idx.unsqueeze(0), global_step=step)

    def save_epoch(self, epoch: int, rt: dict, train_render: torch.Tensor):
        if (epoch + 1) % self.cfg.log_interval_im != 0 and epoch != 0:
            return

        if self.cfg.log:
            idx_list = torch.arange(15)
            fig_dir = self.out_path / "figure" / f"epoch_{epoch + 1}"
            os.makedirs(fig_dir, exist_ok=True)
        else:
            idx_list = torch.randperm(self.cfg.batch_size)[:5]
            fig_dir = None


        def _norm(x):
            x = x.numpy().transpose(1, 2, 0)
            x -= x.min()
            if x.max() > 0:
                x /= x.max()
            return (x * 255).astype(np.uint8)


        grid = torchvision.utils.make_grid(train_render[idx_list])
        ndarr = (
            grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        )
        Image.fromarray(ndarr).save(self.out_path / "images" / f"epoch_{epoch + 1}.png")


def loop(cfg):
    out_path = pathlib.Path(cfg["output_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "config.yml", "w") as f:
        yaml.dump(cfg, f)
    cfg = EasyDict(cfg)
    print("Output directory", cfg.output_path)

    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)

    video = Video(cfg.output_path)
    glctx = dr.RasterizeGLContext()

    resize_method = {
        "cubic": cubic,
        "linear": linear,
        "lanczos2": lanczos2,
        "lanczos3": lanczos3,
    }[cfg.resize_method]

    # misc dirs
    os.makedirs(out_path / "tmp", exist_ok=True)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)
    df = DeepFloydGuidance(cfg, device)

    # prompts
    txt = cfg.text_prompt
    if isinstance(txt, list):
        prompts = [p + ", a 3d rendering" if i != len(txt) - 1 else p for i, p in enumerate(txt)]
    else:
        prompts = [txt + ", a 3d rendering"]
    print("Target text prompt:", txt)
    text_embeds = df.encode_text_2(prompts, batch_size=cfg.batch_size).to(device)
    prompt_num = len(prompts)

    # mesh prep
    if cfg.retriangulate:
        ms.meshing_isotropic_explicit_remeshing()
    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    ms.save_current_mesh(str(out_path / "tmp" / "mesh.obj"))
    load_mesh = obj.load_obj(str(out_path / "tmp" / "mesh.obj"))
    load_mesh = mesh.unit_size(load_mesh)
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(out_path / "tmp" / "mesh.obj"), save_vertex_color=False)

    tex_map  = texture.create_trainable(np.random.uniform(size=[512, 512, 3], low=0.0, high=1.0), [512, 512], True)
    norm_map = texture.create_trainable(np.array([0, 0, 1]), [512, 512], True)
    spec_map = texture.create_trainable(np.array([0, 0, 0]), [512, 512], True)

    load_mesh = mesh.Mesh(
        material={"bsdf": cfg.bsdf, "kd": tex_map, "ks": spec_map, "normal": norm_map},
        base=load_mesh,
    )

    # local deformation stub => to be implemented
    if cfg.local_def:
        data = load_ply(cfg.local_sel)
        v_weight_local = data["vertex_selection"]
        f_mat = load_mesh.t_pos_idx.cpu().numpy()
        f_weight_local = (v_weight_local[f_mat].mean(axis=1) == 1).astype(np.float32)
        f_indicator_local = np.squeeze(f_weight_local > 0, 1)

    jac_src = SourceMesh.SourceMesh(0, str(out_path / "tmp" / "mesh.obj"), {}, 1, ttype=torch.float)
    jac_src.load(); jac_src.to(device)
    with torch.no_grad():
        gt_jac = jac_src.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    if cfg.local_def:
        gt_jac[:, f_indicator_local] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    gt_jac.requires_grad_(True)

    opt = torch.optim.Adam([gt_jac], lr=cfg.lr)
    background = torch.tensor(cfg.background, device=device)

    cams = torch.utils.data.DataLoader(
        CameraBatch(
            cfg.train_res,
            [cfg.dist_min, cfg.dist_max],
            [cfg.azim_min, cfg.azim_max],
            [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
            [cfg.fov_min, cfg.fov_max],
            cfg.aug_loc,
            cfg.aug_light,
            cfg.aug_bkg,
            cfg.batch_size,
            rand_solid=True,
        ),
        cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    for t in ["final", "final_texture", "mesh_log", "best_clip", "best_total"]:
        os.makedirs(out_path / f"mesh_{t}", exist_ok=True)
    for d in ["n_vert", "images", "grads", "jacobians"]:
        os.makedirs(out_path / d, exist_ok=True)

    tb = SummaryWriter(str(out_path / "logs"))
    vis = Visualizer(out_path, cfg, tb)

    rot = 0.0
    for epoch in tqdm(range(cfg.epochs), leave=False):
        # verts from jacobians
        n_vert = jac_src.vertices_from_jacobians(gt_jac).squeeze()

        # blur textures
        def _blur(tex):
            return kornia.filters.gaussian_blur2d(tex.permute(0, 3, 1, 2), (7, 7), (3, 3)).permute(0, 2, 3, 1).contiguous()
        ready_kd = texture.Texture2D(_blur(load_mesh.material["kd"].data))
        ready_ks = texture.Texture2D(_blur(load_mesh.material["ks"].data))
        ready_nm = texture.Texture2D(_blur(load_mesh.material["normal"].data))
        kd_flat  = texture.Texture2D(torch.full_like(ready_kd.data, 0.5))

        m = mesh.Mesh(n_vert, load_mesh.t_pos_idx, material={"bsdf": cfg.bsdf, "kd": kd_flat, "ks": ready_ks, "normal": ready_nm}, base=load_mesh)
        scene = create_scene([m.eval()], sz=512)
        scene = mesh.compute_tangents(mesh.auto_normals(scene))

        # tb mesh
        if (epoch + 1) % cfg.log_interval_im == 0 or epoch == 0:
            cam_p = get_camera_params(cfg.log_elev, rot, cfg.log_dist, cfg.log_res, cfg.log_fov)
            rot += 1
            vis.log_mesh(epoch, scene, cam_p, video, glctx, device)

        # adapt dist
        if cfg.adapt_dist and epoch > 0:
            with torch.no_grad():
                vp = m.v_pos
                vp -= (vp.amin(0) + vp.amax(0)) / 2
                mult = torch.cat([vp.amin(0), vp.amax(0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        cam_batch = next(iter(cams))
        for k in cam_batch:
            cam_batch[k] = cam_batch[k].to(device)
        final_m = scene.eval(cam_batch)
        train_r = render.render_mesh(
            glctx, final_m, cam_batch["mvp"], cam_batch["campos"], cam_batch["lightpos"], cfg.light_power, cfg.train_res, spp=1, num_layers=1, msaa=False, background=torch.broadcast_to(background, [1, cfg.log_res, cfg.log_res, 3])
        ).permute(0, 3, 1, 2)
        train_r = resize(train_r, out_shape=(224, 224), interp_method=resize_method)

        opt.zero_grad()
        for _ in range(cfg.accum_iter):
            if cfg.score == "ActvnReplace":
                rt = df.ActvnReplace(torch.cat([train_r] * (prompt_num + 1)), text_embeds, cfg.modified_cfg, prompt_num=prompt_num, controller=None, attn_ctrl_alphas=cfg.attn_ctrl_alphas)
            else:
                rt = df.SDS(train_r, text_embeds, controller=None)

            img_loss = rt["loss_sds"]
            jac_reg = ((gt_jac - torch.eye(3, device=device)) ** 2).mean()
            tb.add_scalar("jacobian_regularization", jac_reg, global_step=epoch)
            tb.add_scalar("image_loss", img_loss, global_step=epoch)
            (jac_reg * cfg.regularize_jacobians_weight + img_loss * cfg.image_weight).backward(retain_graph=True)
            if cfg.local_def:
                gt_jac.grad[:, f_indicator_local] = 0
            opt.step()

        vis.save_epoch(epoch, rt, train_r)

        if (epoch + 1) % cfg.log_interval_im == 0 or epoch == 0:
            obj.write_obj(str(out_path / "mesh_final"), m.eval())
            np.save(out_path / "jacobians" / f"jacobians_epoch_{epoch + 1}.npy", gt_jac.detach().cpu().numpy())

        del rt["grad"]

    video.close()
    obj.write_obj(str(out_path / "mesh_final"), m.eval())
