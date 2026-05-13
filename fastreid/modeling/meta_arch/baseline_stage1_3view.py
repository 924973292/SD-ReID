# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

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
                       EulerDiscreteScheduler, PNDMScheduler)
from fastreid.modeling.sd import (UNet, VariationalAutoencoder, Inverse_Sampling, ConditionLearner, MemoryBank_base)


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class Baseline_stage1_3view(nn.Module):
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
        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

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

        memory_bank = MemoryBank_base(cfg.MODEL.SDMODEL.VIEW_NUM, cfg.MODEL.SDMODEL.MOEMNTUM)

        return {
            'backbone': backbone,
            'heads': heads,
            'view_heads': view_heads,

            'memory_bank': memory_bank,

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

        view1_index = [index for index, content in enumerate(view) if content == 0]
        view2_index = [index for index, content in enumerate(view) if content == 1]
        view3_index = [index for index, content in enumerate(view) if content == 2]

        features, view_token, ID_features = self.backbone(images, camids)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            temp = torch.zeros((targets.shape[0])).long().to(targets.device)
            # temp[view1_index] = 1
            temp[view2_index] = 1
            temp[view3_index] = 2
            targets_view = temp

            # 为了构建memory bank
            # _ = self.mm(view_token, view).squeeze().unsqueeze(1)
            self.mm(view_token, ID_features, features, view, imgids, stage_condition=1)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            # view_outputs = self.view_heads(view_token, targets_view)

            view_outputs = self.view_heads(view_token, targets_view)

            # # for vis
            # center0 = self.mm.view_centers[0].cpu().detach().numpy()
            # center1 = self.mm.view_centers[1].cpu().detach().numpy()
            #
            # fea0 = view_token[view == 0][:100].squeeze().cpu().detach().numpy()
            # fea1 = view_token[view == 1][:100].squeeze().cpu().detach().numpy()
            #
            # self.cal(fea0, center0, "cos_view0.png", a=0.3, b=0.9, d=1.2,title="",jiange=0.2)
            # self.cal(fea1, center1, "cos_view1.png", a=0.3, b=0.9,d=1.05,title="",jiange=0.2)

            losses = self.losses(outputs, view_outputs, targets, targets_view)

            return losses
        else:
            outputs = self.heads(features)
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


    def cal(self,features,center_feature,path,a,b,c=0,d=1,title="",jiange=1):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import spatial
        from scipy.stats import gaussian_kde

        # 计算余弦相似度
        cosine_similarities = 1 - spatial.distance.cdist(features, center_feature, 'cosine').flatten()

        # 使用 gaussian_kde 进行核密度估计
        kde = gaussian_kde(cosine_similarities)
        kde.set_bandwidth(bw_method=0.2)  # 设置带宽，控制平滑程度

        # 生成用于绘制的密集的x值
        x_vals = np.linspace(0, 1, 1000)  # 从0到1生成x值
        # 计算这些x值的密度
        density = kde(x_vals)

        # 限制核密度估计的范围，使其在0.3到0.6之外平滑过渡到0
        density = np.clip((x_vals - a) / (b - a), 0, 1) * density

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, density, label='KDE', color='white')

        # 填充图线下区域为绿色
        plt.fill_between(x_vals, density, color='green', alpha=0.5, label='Density Fill')

        # 设置横坐标的显示范围
        plt.xlim(a, b)
        plt.ylim(c, d)

        y_ticks = np.arange(c, d, jiange)  # 从0到10，间隔为1
        plt.yticks(y_ticks)

        plt.xticks(fontsize=18)
        # 设置y轴刻度标签的字体大小
        plt.yticks(fontsize=18)

        # 添加辅助的框线
        plt.grid(True, color='gray')

        # 添加图例和标题
        plt.title(title, fontsize=16, weight='bold')
        plt.xlabel('Cosine Similarity', fontsize=24, weight='bold')
        plt.ylabel('Density', fontsize=24, weight='bold')

        # 保存图片
        plt.savefig(path, dpi=500, bbox_inches='tight')

    def flops(self):
        from fvcore.nn.jit_handles import elementwise_flop_counter
        from fvcore.nn import flop_count
        import copy

        # shape = self.__input_shape__[1:]
        # if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
        #     shape = (3, self.image_size[0], self.image_size[1])
        # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = self.give_supported_ops()
        # model = copy.deepcopy(self)
        model = self
        model.cuda().eval()
        # input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input = {"RGB": input_r, "NI": input_n, "TI": input_t}

        # input:{'images':[B,3,256,128],torch.float32,
        #       'targets':[B,],torch.int64,
        #       'camids':[B,],torch.int64,
        #       'viewids':[B,77],torch.float32
        #       }
        B = 2
        input_img = torch.randint(0, 35, (B, 3, 256, 128), device=next(model.parameters()).device).to(torch.float32)
        input_t = torch.randint(1, 807, (B,), device=next(model.parameters()).device).to(torch.int64)
        input_c = torch.randint(0, 2, (B,), device=next(model.parameters()).device).to(torch.int64)
        # input_v = torch.randint(1,3970,(1,77), device=next(model.parameters()).device).to(torch.float32)
        input_v = torch.randint(0, 2, (B,), device=next(model.parameters()).device).to(torch.int64)
        input = {'images': input_img, 'targets': input_t, 'camids': input_c, 'viewids': input_v}

        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        logger.info("Drop_path is excluded from this FLOPs estimate because it is disabled during testing.")
        del model, input
        return sum(Gflops.values()) * 1e9 / B


    def give_supported_ops(self,):
        from fvcore.nn.jit_handles import elementwise_flop_counter

        return{
                "aten::silu": elementwise_flop_counter(0, 1),
                "aten::gelu": elementwise_flop_counter(0, 1),
                "aten::neg": elementwise_flop_counter(0, 1),
                "aten::exp": elementwise_flop_counter(0, 1),
                "aten::flip": elementwise_flop_counter(0, 1),
                "aten::mul": elementwise_flop_counter(0, 1),
                "aten::div": elementwise_flop_counter(0, 1),
                "aten::softmax": elementwise_flop_counter(0, 2),
                "aten::sigmoid": elementwise_flop_counter(0, 1),
                "aten::add": elementwise_flop_counter(0, 1),
                "aten::add_": elementwise_flop_counter(0, 1),
                "aten::radd": elementwise_flop_counter(0, 1),
                "aten::sub": elementwise_flop_counter(0, 1),
                "aten::sub_": elementwise_flop_counter(0, 1),
                "aten::rsub": elementwise_flop_counter(0, 1),
                "aten::mul_": elementwise_flop_counter(0, 1),
                "aten::rmul": elementwise_flop_counter(0, 1),
                "aten::div_": elementwise_flop_counter(0, 1),
                "aten::rdiv": elementwise_flop_counter(0, 1),
                "aten::cumsum": elementwise_flop_counter(0, 1),
                "aten::ne": elementwise_flop_counter(0, 1),
                "aten::silu_": elementwise_flop_counter(0, 1),
                "aten::dropout_": elementwise_flop_counter(0, 1),
                "aten::log_softmax": elementwise_flop_counter(0, 2),
                "aten::argmax": elementwise_flop_counter(0, 1),
                "aten::one_hot": elementwise_flop_counter(0, 1),
                "aten::flatten": elementwise_flop_counter(0, 0),
                "aten::unflatten": elementwise_flop_counter(0, 0),
                "aten::mean": elementwise_flop_counter(1, 0),
                "aten::sum": elementwise_flop_counter(1, 0),
                "aten::abs": elementwise_flop_counter(0, 1),
                "aten::tanh": elementwise_flop_counter(0, 1),
                "aten::relu": elementwise_flop_counter(0, 1),
                "aten::where": elementwise_flop_counter(0, 1),
                "aten::le": elementwise_flop_counter(0, 1),
                "aten::topk": elementwise_flop_counter(1, 1),
                "aten::sort": elementwise_flop_counter(1, 1),
                "aten::argsort": elementwise_flop_counter(1, 1),
                "aten::scatter": elementwise_flop_counter(1, 1),
                "aten::gather": elementwise_flop_counter(1, 1),
                "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
            }