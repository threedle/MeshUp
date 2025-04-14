import os
import yaml
import torch
import random
import argparse
import numpy as np
from loop import loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='./base_config.yml')
    parser.add_argument('--output_path', help='Output directory (will be created)', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--gpu', help='GPU index', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--seed', help='Random seed', type=int, default=argparse.SUPPRESS)
    #parameter for the figures
    parser.add_argument('--log',  action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--no-log',  action='store_false', default=argparse.SUPPRESS, dest='log')
    
    parser.add_argument('--text_prompt', help='Target text prompt', type=str, nargs='+', default=argparse.SUPPRESS)
    # Mesh
    parser.add_argument('--mesh', help='Path to input mesh', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--retriangulate', help='Use isotropic remeshing', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--texture', help='Path to texture', type=str, default=argparse.SUPPRESS)

    # Locality => this capability is to be announced
    parser.add_argument('--local_weight', help='Enables local weighting for vertices', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--no-local_weight', help='Enables local weighting for vertices', action='store_false', default=argparse.SUPPRESS, dest='local_weight')
    parser.add_argument('--target_vertice', type=str, help='Enables local weighting for vertices', default=argparse.SUPPRESS, nargs='+')
    parser.add_argument('--local_def', help='Enables local deformation', type=int, default=0, choices=[0, 1])

    # Render settings
    parser.add_argument('--bsdf', help='Render technique', type=str, default=argparse.SUPPRESS, choices=['diffuse', 'pbr'])

    # Hyper-parameters
    parser.add_argument('--lr', help='Learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--epochs', help='Number of optimization steps', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--regularize_jacobians_weight', help='Weight for jacobian regularization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--regularize_jacobians_local_weight', help='Weight for local jacobian regularization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--batch_size', help='Number of images rendered at the same time', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--accum_iter', help='Number of images rendered at the same time', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--resize_method', help='Image downsampling/upsampling method', type=str, default=argparse.SUPPRESS, choices=['cubic', 'linear', 'lanczos2', 'lanczos3'])

    ## Camera Parameters ##
    parser.add_argument('--fov_min', help='Minimum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--fov_max', help='Maximum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_min', help='Minimum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_max', help='Maximum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--light_power', help='Light intensity', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_alpha', help='Alpha parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_beta', help='Beta parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_max', help='Maximum elevation angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_min', help='Minimum azimuth angle in degree',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_max', help='Maximum azimuth angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aug_loc', help='Offset mesh from center of image?', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_light', help='Augment the direction of light around the camera', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_bkg', help='Augment the background', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--adapt_dist', help='Adjust camera distance to account for scale of shape', type=int, default=argparse.SUPPRESS, choices=[0, 1])

    #diffusion
    parser.add_argument('--model_size', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--dtype', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--cpu_offload', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--no-cpu_offload', action='store_false', default=argparse.SUPPRESS, dest='cpu_offload')
    parser.add_argument('--score', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--replace_alpha', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--modified_cfg', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--no-modified_cfg', action='store_false', default=argparse.SUPPRESS)

    # Logging
    parser.add_argument('--log_interval', help='Interval for logging, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_interval_im', help='Interval for logging renders image, every X epochs',  type=int, default=argparse.SUPPRESS)

    #argmuments used for Blending 
    #this can be a list
    parser.add_argument('--attn_ctrl', action='store_true', help='Turn on blending capabilities', default=argparse.SUPPRESS)
    parser.add_argument('--attn_ctrl_alphas', type=float, nargs='+', default=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]

    print(yaml.dump(cfg, default_flow_style=False))
    random.seed(cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True

    loop(cfg)
    print('Done')

if __name__ == '__main__':
    main()

