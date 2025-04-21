# MeshUp

[Hyunwoo Kim](https://hywkim-brian.github.io/site/), [Itai Lang](https://itailang.github.io/), [Noam Aigerman](https://noamaig.github.io/), [Thibault Groueix](https://imagine.enpc.fr/~groueixt/), [Vladimir G. Kim](http://www.vovakim.com/), and [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)*

<a href="https://threedle.github.io/MeshUp/"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A//threedle.github.io/MeshUp" height=22></a>
<a href="https://arxiv.org/abs/2408.14899"><img src="https://img.shields.io/badge/arXiv-MeshUp-b31b1b.svg" height=22></a>

![](https://raw.githubusercontent.com/threedle/MeshUp/docs/data/concatenated_video.gif)

### Abstract
*We propose MeshUp, a technique that deforms a 3D mesh towards multiple target
concepts, and intuitively controls the region where each concept is expressed.
Conveniently, the concepts can be defined as either text queries, e.g., "a dog"
and "a turtle," or inspirational images, and the local regions can be selected
as any number of vertices on the mesh. We can effectively control the influence
of the concepts and mix them together using a novel score distillation
approach, referred to as the Blended Score Distillation (BSD). BSD operates on
each attention layer of the denoising U-Net of a diffusion model as it extracts
and injects the per-objective activations into a unified denoising pipeline
from which the deformation gradients are calculated. To localize the expression
of these activations, we create a probabilistic Region of Interest (ROI) map on
the surface of the mesh, and turn it into 3D-consistent masks that we use to
control the expression of these activations. We demonstrate the effectiveness
of BSD empirically and show that it can deform various meshes towards multiple
objectives.*

## Overview

MeshUp is designed to enable flexible and intuitive deformation of 3D meshes. Its key innovations include:

- **Multi-target Deformations:** Simultaneously adapt regions of a mesh to different target concepts.
- **Blended Score Distillation (BSD):** Leverage activations from multiple attention layers to guide mesh deformations.
- **Local Control:** Use selectable vertices to precisely control where each concept is applied (To be within one week!)

## Tested Environment

- **GPU:** 48 GB A40 GPU
- **CUDA:** 12.1
- **Python:** 3.10

### Setup environemnt
First create the conda environment:
```
conda create -n "meshup" python=3.10
```
and activate it with:
```
conda activate meshup
```
first, make sure you have the correct pip version
```
conda install pip=23.3.2
```
install torch dependencies (specify the cuda version according to your specs)
```
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
then run
```
conda install -y -c conda-forge igl
pip install -r requirements.txt
```
then install nvdiffrast from source
```
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```
### Login to Hugging Face (to use DeepFloyd IF w/ Diffusers)
Instructions from [DeepFloyd IF](https://github.com/deep-floyd/IF):
1) If you do not already have one, create a [Hugging Face account](https://huggingface.co/join)
2) Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3) Log in to Hugging face locally. First install `huggingface_hub`
```
pip install huggingface_hub --upgrade
```
run the login function in a python shell
```
from huggingface_hub import login

login()
```
and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

## Reproduce paper results
```
python main.py --no-cpu_offload --config ./configs/base_config.yml  --mesh ./meshes/hound.obj --output_path ./outputs/hippo --model_size XL --dtype float16 --score SDS --text_prompt "a hippo"
python main.py --no-cpu_offload --config ./configs/base_config.yml  --mesh ./meshes/hound.obj --output_path ./outputs/dachshund --model_size XL --dtype float16 --score SDS  --text_prompt "a dachshund"
python main.py --no-cpu_offload --config ./configs/base_config.yml  --mesh ./meshes/hound.obj --output_path ./outputs/frog --model_size XL --dtype float16 --score SDS --text_prompt "a frog"
python main.py --no-cpu_offload --config ./configs/base_config.yml  --mesh ./meshes/hound.obj --output_path ./outputs/hippo0.4_frog0.4_dachshound0.2 --model_size XL --dtype float16 --score ActvnReplace --attn_ctrl_alphas 0.4 0.4 0.2 --text_prompt "a hippo" "a frog" "a dachshound" ""
python main.py --no-cpu_offload --config ./configs/base_config.yml  --mesh ./meshes/hound.obj --output_path ./outputs/hippo0.6_frog0.2_dachshound0.2 --model_size XL --dtype float16 --score ActvnReplace --attn_ctrl_alphas 0.6 0.2 0.2 --text_prompt "a hippo" "a frog" "a dachshound" ""
```
Running the following commands will create an `./outputs` directory in your working directory, as specified by the `--output_path` argument.

You can check out the results in:

- `{output_path}/images/` for rendered images  
- `{output_path}/images/mesh_final/` for the exported mesh

<img src="https://raw.githubusercontent.com/threedle/MeshUp/docs/data/hound_deformation.png"/>

## Run your own examples
You can either change the configuration file in configs/base_config.yml or check out the various command line arguments you can apply in main.py.

## TODO
1) Code for memory optimization
2) Code for local optimization

## Acknowledgements
This code base is originally based off of [TextDeformer](https://github.com/threedle/TextDeformer) and [NeuralJacobianFields](https://github.com/ThibaultGROUEIX/NeuralJacobianFields). Thanks to their awesome works!
## Citation
```
@misc{kim2025meshupmultitargetmeshdeformation,
      title={MeshUp: Multi-Target Mesh Deformation via Blended Score Distillation}, 
      author={Hyunwoo Kim and Itai Lang and Noam Aigerman and Thibault Groueix and Vladimir G. Kim and Rana Hanocka},
      year={2025},
      eprint={2408.14899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.14899}, 
}
```
