# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .pcb import PCB
from .moco import MoCo
from .distiller import Distiller
from .baseline_multi_view import Baseline_multiview
from .baseline_clip_visual import Baseline_clip_visual


from .baseline_stage1 import Baseline_stage1
from .baseline_stage2_old import Baseline_stage2_old
from .baseline_stage1_3view import Baseline_stage1_3view
from .baseline_stage2_old_3view_2 import Baseline_stage2_old_3view_2
