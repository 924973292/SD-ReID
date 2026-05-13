import torch
from torch import nn
import copy
from fastreid.modeling.backbones.clip.clip import tokenize


class Inverse_Sampling(nn.Module):
    def __init__(self, cfg, unet):
        super().__init__()
        self.unet = unet

        self.timesteps = cfg.TEST.SDMODEL.NUM_INFERENCE_STEPS

        self.timesteps_train = 1000

        self.down_block_guidance_scale = cfg.TEST.SDMODEL.DOWN_BLOCK_GUIDANCE_SCALE
        self.full_guidance_scale = cfg.TEST.SDMODEL.FULL_GUIDANCE_SCALE
        self.inverse_type = cfg.TEST.SDMODEL.INVERSE_TYPE

        self.generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generator.manual_seed(cfg.MODEL.SEED)

    def forward(
        self,
        noise_scheduler,
        weight_dtype,
        c_new,
        noisy_latents,
        down_block_additional_residuals=None,
        inverse_noise_scheduler=None,
        timesteps_train=None,
        switch=None,
    ):
        bsz = noisy_latents.shape[0]
        type = self.inverse_type

        if type != "inverse":
            noise_scheduler.set_timesteps(self.timesteps)

        if type == "uc_down_full":
            c_new = torch.cat([torch.zeros_like(c_new), torch.zeros_like(c_new), c_new])

            down_block_additional_residuals = [
                torch.cat([torch.zeros_like(sample), sample, sample]).to(
                    dtype=weight_dtype
                )
                for sample in down_block_additional_residuals
            ]
            for t in noise_scheduler.timesteps:
                inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents], dim=0)
                inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
                noise_pred = self.unet(
                    sample=inputs,
                    timestep=t,
                    encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(
                        down_block_additional_residuals
                    ),
                )

                noise_pred_uc, noise_pred_down, noise_pred_full = noise_pred.chunk(3)

                noise_pred = (
                    noise_pred_uc
                    + self.down_block_guidance_scale * (noise_pred_down - noise_pred_uc)
                    + self.full_guidance_scale * (noise_pred_full - noise_pred_down)
                )

                noisy_latents = noise_scheduler.step(
                    noise_pred, t, noisy_latents, generator=self.generator
                )[0]
        elif type == "uc_full":
            c_new = torch.cat([torch.zeros_like(c_new), c_new])
            down_block_additional_residuals = [
                torch.cat([sample, sample], dim=0)
                for sample in down_block_additional_residuals
            ]

            for t in noise_scheduler.timesteps:
                inputs = torch.cat([noisy_latents, noisy_latents], dim=0)
                inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
                if switch == 1:
                    length = len(down_block_additional_residuals)
                    up_block_additional_residuals = down_block_additional_residuals[
                        length // 2 :
                    ]
                    down_block_additional_residuals = down_block_additional_residuals[
                        : length // 2
                    ]

                    noise_pred = self.unet(
                        sample=inputs,
                        timestep=t,
                        encoder_hidden_states=c_new,
                        up_block_additional_residuals=copy.deepcopy(
                            up_block_additional_residuals
                        ),
                        down_block_additional_residuals=copy.deepcopy(
                            down_block_additional_residuals
                        ),
                    )

                noise_pred_uc, noise_pred_full = noise_pred.chunk(2)
                noise_pred = noise_pred_uc + self.full_guidance_scale * (
                    noise_pred_full - noise_pred_uc
                )
                noisy_latents = noise_scheduler.step(
                    noise_pred, t, noisy_latents, generator=self.generator
                )[0]

        elif type == "down_full":
            down_block_additional_residuals = [
                torch.cat([sample, sample]).to(dtype=weight_dtype)
                for sample in down_block_additional_residuals
            ]

            for t in noise_scheduler.timesteps:
                inputs = torch.cat([noisy_latents, noisy_latents], dim=0)
                inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
                noise_pred = self.unet(
                    sample=inputs,
                    timestep=t,
                    encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(
                        down_block_additional_residuals
                    ),
                ).sample

                noise_pred_down, noise_pred_full = noise_pred.chunk(2)
                noise_pred = noise_pred_down + self.full_guidance_scale * (
                    noise_pred_full - noise_pred_down
                )
                noisy_latents = noise_scheduler.step(
                    noise_pred, t, noisy_latents, generator=self.generator
                )[0]

        elif type == "full":
            c_new = c_new
            for t in noise_scheduler.timesteps:
                inputs = noisy_latents
                inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
                noise_pred = self.unet(
                    sample=inputs,
                    timestep=t,
                    encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(
                        down_block_additional_residuals
                    ),
                )

                noisy_latents = noise_scheduler.step(
                    noise_pred, t, noisy_latents, generator=self.generator
                )[0]

        elif type == "inverse":
            inverse_noise_scheduler.set_timesteps(self.timesteps)

            down_block_additional_residuals = [
                sample[:bsz] for sample in down_block_additional_residuals
            ]

            for t in inverse_noise_scheduler.timesteps:
                inputs = noisy_latents
                noise_pred = self.unet(
                    sample=inputs,
                    timestep=t,
                    encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(
                        down_block_additional_residuals
                    )
                    if down_block_additional_residuals
                    else None,
                )
                noisy_latents = inverse_noise_scheduler.step(
                    noise_pred, t, noisy_latents, generator=self.generator
                )[0]
            return noisy_latents

        noise_scheduler.set_timesteps(
            self.timesteps_train if timesteps_train is None else timesteps_train
        )

        return noisy_latents
