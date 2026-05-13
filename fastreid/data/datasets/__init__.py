# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """Registry for SD-ReID benchmark datasets."""

from .cargo import CARGO, CARGO_AA, CARGO_GG, CARGO_AG
from .g2apsreid import G2APS_ReID, G2APS_ReID_A2G, G2APS_ReID_G2A
from .agreidv1 import AG_ReID_v1, AG_ReID_v1_AG, AG_ReID_v1_GA
from .agreidv2 import AG_ReID_v2_WA, AG_ReID_v2_AW, AG_ReID_v2_AC, AG_ReID_v2_CA, AG_ReID_v2
from .lagper import LAGPeR, LAGPeR_A2G, LAGPeR_G2A, LAGPeR_G2AG

__all__ = [key for key in globals().keys() if "builtin" not in key and not key.startswith("_")]
