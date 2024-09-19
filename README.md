# MeshUp

*[Hyunwoo Kim], [Itai Lang](https://itailang.github.io/), [Noam Aigerman](https://noamaig.github.io/), [Thibault Groueix](https://imagine.enpc.fr/~groueixt/), [Vladimir G. Kim](http://www.vovakim.com/) [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)*

<a href="https://threedle.github.io/MeshUp/"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A//threedle.github.io/MeshUp" height=22></a>
<a href="https://arxiv.org/abs/2408.14899"><img src="https://img.shields.io/badge/arXiv-MeshUp-b31b1b.svg" height=22></a>


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

The code will be released soon!
