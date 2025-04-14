#https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/deep_floyd_guidance.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline
from diffusers.utils.import_utils import is_xformers_available
from ptp_utils import register_activation_control, unregister_attention_control, save_forward

class DeepFloydGuidance(nn.Module):
    def __init__(self, cfg, device='cuda', t_range=[0.02, 0.98]):
        super().__init__()
        # Create model
        #torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = device
        if cfg.model_size == "XL":
            model_name = "DeepFloyd/IF-I-XL-v1.0"
        elif cfg.model_size == "L":
            model_name = "DeepFloyd/IF-I-L-v1.0"
        elif cfg.model_size == "M":
            model_name = "DeepFloyd/IF-I-M-v1.0"
        if cfg.dtype == "float32":
            dtype = torch.float32
            variant="fp32"
        elif cfg.dtype == "float16":
            dtype = torch.float16
            variant="fp16"
        self.model_name = model_name
        self.variant = variant
        self.dtype = dtype
        self.pipe = IFPipeline.from_pretrained(
            model_name,
            variant=variant, 
            torch_dtype=dtype,
            #text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        self.weights_dtype = dtype
        #self.pipe.enable_xformers_memory_efficient_attention()
        if cfg.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        #self.pipe.enable_attention_slicing(1)
        self.pipe.unet.to(memory_format=torch.channels_last)
        #p.requires_grad_(False)
        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )



    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode_text(self, prompt, negative_prompt=None, batch_size=1):
        pos_embeds, neg_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        pos_embeds = torch.cat([pos_embeds] * batch_size, dim=0)
        neg_embeds = torch.cat([neg_embeds] * batch_size, dim=0)
        prompt = torch.cat([pos_embeds, neg_embeds])
        return prompt
    
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode_text_2(self, prompt, negative_prompt=None, batch_size=1, return_mask=False):
        new_prompt=[]
        max_length=77
        for text in prompt:
            new_prompt.extend([text]*batch_size)
        if return_mask:
            mask = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )['attention_mask']
        pos_embeds, neg_embeds = self.pipe.encode_prompt(new_prompt, negative_prompt=negative_prompt)
        pos_embeds = torch.cat([pos_embeds], dim=0)
        neg_embeds = torch.cat([neg_embeds], dim=0)[:(pos_embeds.shape[0]//len(prompt))]
        prompt = torch.cat([pos_embeds, neg_embeds])
        if return_mask:
            return prompt, mask
        return prompt
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents, t, text_embeds):
        input_dtype = latents.dtype
        return self.pipe.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            text_embeds.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def SDS(
        self,
        rgb,
        text_embeds,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        #rgb_BCHW = rgb.permute(0, 3, 1, 2)
        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb, (64, 64), mode="bilinear", align_corners=False
        )
        #we can potentially do this later
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            int((self.max_step + 1)),
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                text_embeds,
            )  # (B, 6, 64, 64)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        }
   
        
    def ActvnReplace(
        self,
        rgb,
        text_embeds,
        iter=None,
        modified_cfg=False, 
        prompt_num=2,
        attn_ctrl_alphas=None, 
        **kwargs,
    ):
        batch_size = rgb.shape[0] // (prompt_num+1)
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb, (64, 64), mode="bilinear", align_corners=False
        )
        #we can potentially do this later
        t_mask = torch.randint(
            int(self.min_step),
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        t_mask = torch.cat([t_mask]*prompt_num)# timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t_ = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        t = torch.cat([t_]*prompt_num)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # we simply cache the previous forward pass function so we can recover
            save_forward(self.pipe)
            register_activation_control(self.pipe, None, prompt_num=prompt_num, replace_alpha=attn_ctrl_alphas) 
            noise_ = torch.randn_like(latents[:(latents.shape[0]//(prompt_num+1))])  # TODO: use torch generator
            noise = torch.cat([noise_]*prompt_num)
            latents_noisy = self.scheduler.add_noise(latents[:batch_size*prompt_num], noise, t)
            latents_noisy_uncond = self.scheduler.add_noise(latents[batch_size*prompt_num:], noise_, t_)
            t_noise_pred_text = self.forward_unet(
                latents_noisy,
                t,
                text_embeds[:batch_size*prompt_num],
            )
            save_forward(self.pipe)
            unregister_attention_control(self.pipe)
            t_noise_pred_uncond= self.forward_unet(
                latents_noisy_uncond,
                t_,
                text_embeds[batch_size*prompt_num:],
            )  # (B, 6, 64, 64)
        t_noise_pred_text = t_noise_pred_text.chunk(prompt_num)[-1]
        t_noise_pred_text, t_predicted_variance = t_noise_pred_text.split(3, dim=1)
        t_noise_pred_uncond, _ = t_noise_pred_uncond.split(3, dim=1)
        t_noise_pred = t_noise_pred_text + self.cfg.guidance_scale * (
            t_noise_pred_text - t_noise_pred_uncond
        )
        w = (1 - self.alphas[t_]).view(-1, 1, 1, 1)
        grad = w * (t_noise_pred - noise_)# * mask[:8]
        grad = torch.nan_to_num(grad)

        latents = latents.chunk(prompt_num+1)[-2]
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        } 

   