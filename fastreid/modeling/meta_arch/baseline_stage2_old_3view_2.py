# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

import timm

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.layers import trunc_normal_

from diffusers import (DDIMInverseScheduler, DDIMScheduler, DDPMScheduler,
                       EulerDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler,
                       UNet2DConditionModel)
from fastreid.modeling.sd import (UNet, VariationalAutoencoder, Inverse_Sampling, ConditionLearner, MemoryBank_base)



@META_ARCH_REGISTRY.register()
class Baseline_stage2_old_3view_2(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            view_heads,

            memory_bank,
            noise_scheduler,
            inverse_noise_scheduler,
            unet,
            inverse_sampling,
            cubic_sampling,
            condition_learner,
            train_step,

            switch2,

            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self.view_heads = view_heads

        # sd
        self.mm = memory_bank

        self.inverse_noise_scheduler = inverse_noise_scheduler
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.inverse_sampling = inverse_sampling
        self.cubic_sampling = cubic_sampling
        self.condition_learner = condition_learner

        self.train_step = train_step

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.lock_net(self.backbone)
        self.lock_net(self.heads)
        self.lock_net(self.view_heads)
        self.lock_net(self.mm)

        self.switch2 = switch2

    def lock_net(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_tokens'}

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()

        cfg0.MODEL.HEADS.NUM_CLASSES = 3

        view_heads = build_heads(cfg0)
        cfg0 = cfg.clone()

        # new_added
        train_step = cfg.MODEL.SDMODEL.TRAIN_STEP


        if cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.NAME == "euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
            noise_scheduler.config["num_train_timesteps"] = train_step
            noise_scheduler = EulerDiscreteScheduler.from_config(noise_scheduler.config)
        elif cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.NAME == "pndm":
            noise_scheduler = PNDMScheduler.from_pretrained(cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
            noise_scheduler.config["num_train_timesteps"] = train_step
            noise_scheduler = PNDMScheduler.from_config(noise_scheduler.config)
        elif cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.NAME == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained(cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
            noise_scheduler.config["num_train_timesteps"] = train_step
            noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        elif cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.NAME == "ddpm":
            noise_scheduler = DDPMScheduler.from_pretrained(cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
            noise_scheduler.config["num_train_timesteps"] = train_step
            noise_scheduler = DDPMScheduler.from_config(noise_scheduler.config)

        inverse_noise_scheduler = DDIMInverseScheduler(
            num_train_timesteps=noise_scheduler.num_train_timesteps,
            beta_start=noise_scheduler.beta_start,
            beta_end=noise_scheduler.beta_end,
            beta_schedule=noise_scheduler.beta_schedule,
            trained_betas=noise_scheduler.trained_betas,
            clip_sample=noise_scheduler.clip_sample,
            set_alpha_to_one=noise_scheduler.set_alpha_to_one,
            steps_offset=noise_scheduler.steps_offset,
            prediction_type=noise_scheduler.prediction_type,

            # timestep_spacing='leading'
            timestep_spacing=noise_scheduler.timestep_spacing
        )


        unet = UNet(cfg)
        # unet = UNet2DConditionModel.from_pretrained(cfg.MODEL.SDMODEL.UNET_CONFIG.PRETRAINED_PATH)
        # unet.requires_grad_(False)

        inverse_sampling = Inverse_Sampling(cfg, unet)
        condition_learner = ConditionLearner(cfg)

        memory_bank = MemoryBank_base(cfg.MODEL.SDMODEL.VIEW_NUM, cfg.MODEL.SDMODEL.MOEMNTUM)

        switch2 = cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.SWITCH2

        return {
            'backbone': backbone,
            'heads': heads,
            'view_heads': view_heads,

            'memory_bank': memory_bank,
            'noise_scheduler': noise_scheduler,
            'inverse_noise_scheduler': inverse_noise_scheduler,
            'unet': unet,
            'inverse_sampling': inverse_sampling,
            'cubic_sampling': cfg.MODEL.SDMODEL.SCHEDULER_CONFIG.CUBIC_SAMPLING,
            'condition_learner': condition_learner,

            'train_step': train_step,
            'switch2': switch2,

            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'view_id': cfg.MODEL.LOSSES.CE.VIEW_ID,
                        'view_oreg': cfg.MODEL.LOSSES.CE.VIEW_OREG,
                        'view_lambda': cfg.MODEL.LOSSES.CE.VIEW_LAMBDA,
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },

                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        camids = batched_inputs['camids']
        # imgids = [path.rsplit('/', 1)[1][:-4] for path in batched_inputs['img_paths']]
        imgids = None

        view = batched_inputs['viewids']
        # view1_index = [index for index, content in enumerate(view) if content == 'Aerial']
        # view2_index = [index for index, content in enumerate(view) if content == 'Ground']


        with torch.no_grad():
            features, view_token, ID_features = self.backbone(images, camids)

        # 从meomry_bank中获得view
        # view_token_new, ID_features, features = self.mm(None, None, None, view, imgids, stage_condition=2)
        # view_token_new, _, _ = self.mm(None, None, None, view, imgids, stage_condition=2)
        # view_token_new = view_token_new.squeeze().unsqueeze(1)
        view_token_new = view_token.squeeze().unsqueeze(1)

        # view_token_center, _, _ = self.mm(None, None, None, view, imgids, stage_condition=2)
        view_token_center, _, _ = self.mm(None, None, features, view, imgids, stage_condition=2)
        view_token_center = view_token_center.squeeze().unsqueeze(1)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            losses = dict()
            # outputs = self.heads(features, targets)
            # view_outputs = self.view_heads(view_token, targets_view)
            # losses = self.losses(outputs, view_outputs, targets, targets_view)

            # SD
            latents = features.reshape(features.shape[0], 4, 16, 12)

            # features = features.squeeze().unsqueeze(1).repeat(1,4,1)
            # latents = features.reshape(features.shape[0], -1, 32, 24)

            self.noise_scheduler.config.num_train_timesteps = self.train_step

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            if self.cubic_sampling:
                # Cubic sampling to sample a random timestep for each image
                timesteps = torch.rand((bsz,), device='cuda')
                timesteps = (1 - timesteps ** 3) * self.noise_scheduler.config.num_train_timesteps
                timesteps = timesteps.long()
                timesteps = torch.clamp(timesteps, 0, self.noise_scheduler.config.num_train_timesteps - 1)
            else:
                # Uniform sampling to sample a random timestep for each image
                timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (bsz,), device='cuda')

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            weight_type = torch.float32

            c, down_block_additional_residuals = self.condition_learner(ID_features, view_token_new,view_token_center)

            encoder_hidden_states = c.to(dtype=weight_type)
            if down_block_additional_residuals is not None:
                down_block_additional_residuals = [samper.to(dtype=weight_type) for samper in
                                                   down_block_additional_residuals]

            # model_pred = self.unet(
            #     sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
            #     down_block_additional_residuals=down_block_additional_residuals,
            #     up_block_additional_residuals=up_block_additional_residuals)

            if self.switch2 == 0:
                model_pred = self.unet(
                    sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_additional_residuals)
            elif self.switch2 == 1:
                # model_pred = self.unet(
                #     sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                #     up_block_additional_residuals=down_block_additional_residuals)
                length = len(down_block_additional_residuals)
                up_block_additional_residuals = down_block_additional_residuals[length//2:]
                down_block_additional_residuals = down_block_additional_residuals[:length // 2]

                model_pred = self.unet(
                    sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                    up_block_additional_residuals=up_block_additional_residuals,
                    down_block_additional_residuals=down_block_additional_residuals,
                )

            elif self.switch2 == 2:
                model_pred = self.unet(
                    sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                    up_block_additional_residuals=down_block_additional_residuals)

            elif self.switch2 == 4:
                model_pred = self.unet(
                    sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states)

            loss_simple = (noise - model_pred) ** 2
            loss_simple = loss_simple.mean()

            losses['loss_mse'] = loss_simple

            return losses
        else:
            # features, view_token, ID_features = self.backbone(images, camids)
            view_token_new, _, _ = self.mm(None, None, None, view, imgids, stage_condition=2)
            view_token_new = view_token_new.squeeze().unsqueeze(1)

            # # SD
            # bsz = images.shape[0]
            # c, down_block_additional_residuals = self.condition_learner(ID_features, view_token_new, view_token_center)
            # # noisy_latents = torch.randn((bsz, 4, img_size[0] // 8, img_size[1] // 8)).to('cuda')
            #
            # # c = self.condition_learner(ID_features, view_token, view)
            #
            # noisy_latents = torch.randn((bsz, 4, 16, 12)).to('cuda')
            weight_dtype = torch.float32
            bsz = view.shape[0]

            if self.mm.view_num == 2:
                view_token_new1 = view_token_new[:bsz]
                view_token_new2 = view_token_new[bsz:]

                c1, down_block_additional_residuals1 = self.condition_learner(ID_features, view_token_new1,
                                                                              view_token_new1)
                c2, down_block_additional_residuals2 = self.condition_learner(ID_features, view_token_new2,
                                                                              view_token_new2)

                noisy_latents1 = torch.randn((bsz, 4, 16, 12)).to('cuda')
                noisy_latents2 = torch.randn((bsz, 4, 16, 12)).to('cuda')

                latents1 = self.inverse_sampling(self.noise_scheduler, weight_dtype, c1, noisy_latents1,
                                                 down_block_additional_residuals1, self.inverse_noise_scheduler,
                                                 self.train_step, self.switch2).flatten(1)
                latents2 = self.inverse_sampling(self.noise_scheduler, weight_dtype, c2, noisy_latents2,
                                                 down_block_additional_residuals2, self.inverse_noise_scheduler,
                                                 self.train_step, self.switch2).flatten(1)

                outputs = torch.cat([latents1, latents2], dim=1)
                # outputs = latents1

            elif self.mm.view_num == 3:
                view_token_new1 = view_token_new[:bsz]
                view_token_new2 = view_token_new[bsz: bsz*2]
                # view_token_new3 = view_token_new[bsz * 2:]

                c1, down_block_additional_residuals1 = self.condition_learner(ID_features, view_token_new1,
                                                                              view_token_new1)
                c2, down_block_additional_residuals2 = self.condition_learner(ID_features, view_token_new2,
                                                                              view_token_new2)

                # c3, down_block_additional_residuals3 = self.condition_learner(ID_features, view_token_new3,view_token_new3)


                noisy_latents1 = torch.randn((bsz, 4, 16, 12)).to('cuda')
                noisy_latents2 = torch.randn((bsz, 4, 16, 12)).to('cuda')
                # noisy_latents3 = torch.randn((bsz, 4, 16, 12)).to('cuda')

                latents1 = self.inverse_sampling(self.noise_scheduler, weight_dtype, c1, noisy_latents1,
                                                 down_block_additional_residuals1, self.inverse_noise_scheduler,
                                                 self.train_step, self.switch2).flatten(1)
                latents2 = self.inverse_sampling(self.noise_scheduler, weight_dtype, c2, noisy_latents2,
                                                 down_block_additional_residuals2, self.inverse_noise_scheduler,
                                                 self.train_step, self.switch2).flatten(1)
                # latents3 = self.inverse_sampling(self.noise_scheduler, weight_dtype, c3, noisy_latents3,
                #                                  down_block_additional_residuals3, self.inverse_noise_scheduler,
                #                                  self.train_step, self.switch2).flatten(1)

                # outputs = torch.cat([latents1, latents2,latents3], dim=1)
                outputs = torch.cat([latents1, latents2], dim=1)
                # outputs = torch.cat([latents1, latents3], dim=1)


            outputs = torch.cat([self.heads(features), outputs], dim=1)


            # outputs = latents2

            # visual = self.heads(features)
            # #
            # outputs = torch.zeros_like(visual)
            # #
            # outputs[view == 0] = visual[view == 0]
            # outputs[view == 1] = latents2[view == 1]
            #
            # outputs = torch.cat([visual, outputs], dim=1)

            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, outputs_view, gt_labels, view_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']

        # 记得根据情况调整！！！！
        view_cls_outputs = outputs_view['cls_outputs']
        view_pred_features = outputs_view['features']

        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        view_kwargs = self.loss_kwargs.get('ce')
        view_id_flag = view_kwargs.get('view_id')
        view_oreg_flag = view_kwargs.get('view_oreg')
        view_lambda = view_kwargs.get('view_lambda')

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_id'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            if view_id_flag:
                loss_dict['loss_cls_view'] = cross_entropy_loss(
                    view_cls_outputs,
                    view_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale') * view_lambda

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_id'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        # # # calc oreg loss part
        # if view_oreg_flag:
        #     loss_dict['loss_oreg'] = torch.cosine_similarity(pred_features, view_pred_features).abs().mean() * view_lambda

        return loss_dict

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
