import torch
from torch import nn
from fastreid.modeling.backbones.clip.clip import tokenize
from fastreid.modeling.backbones.vision_transformer import Block as AttentionBlock
from fastreid.layers import DropPath, trunc_normal_, to_2tuple

from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers.models.resnet import ResnetBlock2D, Downsample2D
from diffusers.models.attention import BasicTransformerBlock
from fastreid.modeling.backbones.clip.model import Transformer
from safetensors.torch import load_file


# 注意现在的版本只针对两个视角的！！！！！
# 记得回头把参数写到config文件！！


class SEBlock1D(nn.Module):
    def __init__(self, r, in_channel, out_channel):
        super().__init__()
        self.r = r

        self.proj1 = nn.Linear(in_channel, out_channel)

        self.proj2_1 = nn.Linear(in_channel, out_channel // self.r)
        self.proj2_2 = nn.Linear(out_channel // self.r, out_channel)

        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.proj1(x)
        w = self.act2(self.proj2_2(self.act1(self.proj2_1(x))))

        x = x1 * w

        return x.unsqueeze(-1).unsqueeze(-1)


class IDRejector(nn.Module):
    def __init__(self, r, in_channel, out_channel):
        super().__init__()

        self.in_channel = in_channel
        self.out_channels = out_channel
        self.r = r
        self.AttentionBlock = AttentionBlock(
            dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
        )
        self.SEs = nn.ModuleList(
            SEBlock1D(self.r, self.in_channel, c) for c in self.out_channels
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = torch.stack(x, dim=1)
        x = self.AttentionBlock(x)
        x = x.mean(dim=1)

        fea = []
        for SE in self.SEs:
            fea.append(SE(x))
        return fea


class MemoryBank_base(nn.Module):
    def __init__(self, view_num, momentum, dim=768):
        super().__init__()
        self.momentum = momentum
        self.view_num = view_num
        view_centers = [
            nn.Parameter(torch.zeros(1, dim), requires_grad=False)
            for _ in range(self.view_num)
        ]
        self.view_centers = nn.ParameterList(view_centers)

    def update(self, data, vid):
        data_new = data.detach()
        data_new = data_new.squeeze()
        self.view_centers[vid] = (
            self.momentum * data_new.mean(dim=0)
            + (1 - self.momentum) * self.view_centers[vid]
        )

    def forward(
        self, view_token, ID_features, cls_features, viewids, imageids, stage_condition
    ):
        with torch.no_grad():
            if stage_condition == 1:
                viewid = viewids.unique()
                for vid in viewid:
                    self.update(view_token[viewids == vid], vid)

            # 记得修改！！！！
            elif stage_condition == 2:
                bsz = viewids.shape[0]
                if self.training:
                    view_token = [self.view_centers[i] for i in viewids]
                else:
                    view_token = []
                    for i in range(self.view_num):
                        view_token = view_token + [
                            self.view_centers[i] for _ in viewids
                        ]
                return torch.stack(view_token).squeeze(), None, None


class Conv_Pool_Dwonsampler(nn.Module):
    def __init__(self, out_channels, factor):
        super().__init__()

        self.f = factor
        self.out_channels = out_channels
        self.fc = nn.Linear(768, out_channels * 32 * 24 // factor // factor)

    def forward(self, view):
        v = view.flatten(1)
        v = self.fc(v)
        v = v.reshape(v.shape[0], self.out_channels, 32 // self.f, 24 // self.f)

        return v


class TransformerBlock(BasicTransformerBlock):
    def forward(
        self,
        hidden_states,
        query_pos,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice)
                    for hid_slice in norm_hidden_states.chunk(
                        num_chunks, dim=self._chunk_dim
                    )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class ConditionLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.learnable_vector = nn.Parameter(
            torch.randn(
                (
                    1,
                    cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.N_CTX,
                    cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.CTX_DIM,
                )
            )
        )

        self.u_cond_percent = cfg.MODEL.SDMODEL.U_COND_PERCENT
        self.u_cond_down_block_guidance = cfg.MODEL.SDMODEL.U_COND_DOWN_BLOCK_GUIDANCE

        self.in_channel = cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.IN_CHANNEL
        self.out_channels = cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.OUT_CHANNELS

        self.switch = cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.SWITCH

        downsamplers = []
        factor = [2, 4, 8] if len(self.out_channels) < 4 else [2, 4, 8, 2, 4, 8]

        for i in range(len(self.out_channels)):
            in_channels = self.in_channel if i == 0 else self.out_channels[i - 1]
            out_channels = self.out_channels[i]

            downsamplers.append(nn.Conv2d(1, out_channels, factor[i], factor[i]))

        self.downsamplers = nn.ModuleList(downsamplers)

        self.trans = Transformer(768, cfg.MODEL.SDMODEL.CON_LEARNER_CONFIG.DEPTH, 12)

    def forward(self, ID_features, view_token, view_token_center):
        bsz = view_token.shape[0]

        if isinstance(ID_features, list):
            ID_features = torch.stack(ID_features, dim=1)

        if len(ID_features.shape) == 2:
            ID_features = ID_features.unsqueeze(1)

        encoder_hidden_states = ID_features
        if self.switch == 2:
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, view_token_center], dim=1
            )
            c = self.trans(encoder_hidden_states)

        down_block_additional_residuals = []
        view_map_raw = view_token.reshape(bsz, 1, 32, 24)

        for downsampler in self.downsamplers:
            view_map = downsampler(view_map_raw)
            down_block_additional_residuals.append(view_map)

        if self.training:
            u_cond_prop = torch.rand(bsz, 1, 1)
            u_cond_prop = (u_cond_prop < self.u_cond_percent).to(
                dtype=view_token.dtype, device=view_token.device
            )

            u_cond_prop = u_cond_prop.expand(-1, c.shape[1], c.shape[2])
            c = self.learnable_vector.expand(bsz, -1, -1).to(
                dtype=view_token.dtype
            ) * u_cond_prop + c * (1 - u_cond_prop)
        return c, down_block_additional_residuals
