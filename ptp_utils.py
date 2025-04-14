# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch.nn.functional as F
#from utils import pix2patch
import math
import matplotlib.pyplot as plt


def print_attention(attn, num_imgs, dim, title, count):
    img_map = attn[:num_imgs, ...].view(num_imgs, dim, dim)
    fig, axs = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))
    for i, img_tensor in enumerate(img_map):
        axs[i].imshow(img_tensor.detach().cpu(), cmap='gray')
        axs[i].axis('off')
        var = img_tensor[i]
    fig.suptitle(title+ str(count), fontsize=16)
    save_dir = './paper_imgs' 
    import os
    base_filename = 'attn_map'
    counter = 0
    while os.path.exists(os.path.join(save_dir, f'{base_filename}_{counter}.png')):
        counter += 1
    fig.savefig(os.path.join(save_dir, f'{base_filename}_{counter}.png'))
    plt.close(fig)


@torch.no_grad()
def text2image_deepfloyd(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)
    generator = torch.manual_seed(8888)
    prompt_embeds, negative_embeds = model.encode_prompt(prompt)
    image = model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, num_images_per_prompt=1, generator=generator, output_type="pt", num_inference_steps=num_inference_steps).images
    return image 

class LoRALinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

def save_forward(model):
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            #overwrite the __call__ method instead of forward
            net_.save_forward = net_.forward
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

   
def unregister_attention_control(model):
    def unregister_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            #overwrite the __call__ method instead of forward
            net_.forward = net_.save_forward
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = unregister_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += unregister_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += unregister_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += unregister_recr(net[1], 0, "mid")

#main wrapper function for naive blending (without localization)
def register_activation_control(model, controller, control_vertex=None, vp_map=None, replace_alpha=.7, prompt_num=2, random_replace=False):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None):
            αn = len(self.replace_alpha)
            B  = hidden_states.shape[0] // (αn + 1)

            residual = hidden_states                                          
            hidden_states = hidden_states.view(hidden_states.size(0),
                                            hidden_states.size(1),
                                            -1).transpose(1, 2)

            N, L, _ = hidden_states.shape
            attn_mask = self.prepare_attention_mask(attention_mask, L, N, out_dim=4)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states)

            hidden_states = self.group_norm(hidden_states.transpose(1, 2)
                                            ).transpose(1, 2)

            q = self.head_to_batch_dim(self.to_q(hidden_states), out_dim=4)
            k_cross = self.head_to_batch_dim(self.add_k_proj(encoder_hidden_states),
                                            out_dim=4)
            v_cross = self.head_to_batch_dim(self.add_v_proj(encoder_hidden_states),
                                            out_dim=4)

            if self.only_cross_attention:
                k, v = k_cross, v_cross
            else:
                k_self = self.head_to_batch_dim(self.to_k(hidden_states), out_dim=4)
                v_self = self.head_to_batch_dim(self.to_v(hidden_states), out_dim=4)
                k = torch.cat([k_cross, k_self], dim=2)
                v = torch.cat([v_cross, v_self], dim=2)

            q = q.flatten(0, 1)        
            k = k.flatten(0, 1)
            v = v.flatten(0, 1)

            attn = self.get_attention_scores(q, k, None) 
            out  = torch.bmm(attn, v)                     

            out = self.batch_to_head_dim(out)           
            out = self.to_out[1](self.to_out[0](out))     
            out = out.transpose(1, 2).reshape(residual.shape)

            # **Blend replacement branches here** and add residual
            blend = sum(out[i*B:(i+1)*B] * self.replace_alpha[i]
                        for i in range(αn))
            out[B*αn:B*(αn+1)] = blend

            return out + residual
        return forward
  

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
    
    class GlobalRank:
        ranklist = []
        mean_rank = None

    class DummyChild(GlobalRank):
        def append(self, element):
            self.ranklist.append(element)
        def access_rank(self):
            return self.ranklist
        def access_mean(self):
            return self.mean_rank
        def update_mean(self):
            self.mean_rank = torch.stack(self.ranklist).mean(dim=0)
    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet, control_vertex, vp_map, replace_alpha, prompt_num, random_replace):
        if net_.__class__.__name__ == 'Attention':
            net_.rankmethod = DummyChild()
            net_.forward = ca_forward(net_, place_in_unet)
            net_.place_in_unet = place_in_unet
            net_.control_vertex = control_vertex
            net_.vp_map = vp_map
            net_.replace_alpha = replace_alpha
            net_.prompt_num = prompt_num
            net_.random_replace = random_replace 
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet, control_vertex, vp_map, replace_alpha, prompt_num, random_replace)
        return count
    if prompt_num > 2:
        #if prompt number is greater than 2, than replace alpha should be a list of length prompt_number-1
        assert len(replace_alpha) == prompt_num - 1
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down", control_vertex, vp_map, replace_alpha, prompt_num, random_replace)
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up", control_vertex, vp_map, replace_alpha, prompt_num, random_replace)
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid", control_vertex, vp_map, replace_alpha, prompt_num, random_replace)
    controller.num_att_layers = cross_att_count

def collect_mask(unet):
    def collect_mask_(net_, hs):
        if net_.__class__.__name__ == 'Attention':
            try:
                hs.append(net_.mask)
            except:
                print("No mask found")
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                collect_mask_(net__, hs)
    hs = []
    sub_nets = unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            collect_mask_(net[1], hs)
        elif "up" in net[0]:
            collect_mask_(net[1], hs)
        elif "mid" in net[0]:
            collect_mask_(net[1], hs)
    return hs
    


def gather_interpolation_loss(model):
    def gather_interpolation_loss_(net_, loss, count):
        if net_.__class__.__name__ == 'Attention':
            if net_.processor.__class__.__name__ == 'AttnAddedKVProcessor2_0':
                if net_.bottle_neck:
                    loss_ = torch.stack(net_.interpolation_loss)
                    count_add = len(loss_)
                    loss += loss_.sum()
                    count = count + count_add
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                loss, count = gather_interpolation_loss_(net__, loss, count)
        return loss, count
    interpolation_loss = torch.zeros([1]).to(model.unet.device)
    net_count = 0 
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            loss, count = gather_interpolation_loss_(net[1], interpolation_loss, net_count)
            interpolation_loss += loss
            net_count += count
        elif "up" in net[0]:
            loss, count = gather_interpolation_loss_(net[1], interpolation_loss, net_count)
            interpolation_loss += loss
            net_count += count
        elif "mid" in net[0]:
            loss, count = gather_interpolation_loss_(net[1], interpolation_loss, net_count)
            interpolation_loss += loss
            net_count += count
    return interpolation_loss / net_count
    
def get_word_inds(text: str, word_place, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words