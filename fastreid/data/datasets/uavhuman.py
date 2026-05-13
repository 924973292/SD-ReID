# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

__all__ = ['UAVHuman', ]

@DATASET_REGISTRY.register()
class UAVHuman(ImageDataset):
    dataset_dir = 'uavhuman'
    dataset_name = "uavhuman"

    def __init__(self, root='datasets', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)
        self.cam_index = 1

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(UAVHuman, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        data = []
        for img_path in img_paths:

            try:
                match_ps = re.search(r'P(\d+)S', img_path.split('/')[-1])
                match_sg = re.search(r'S(\d+)G', img_path.split('/')[-1])
                pid = int(match_ps.group(1) + match_sg.group(1))
            except:
                match_ds = re.search(r'D(\d+)S', img_path.split('/')[-1])
                match_sg = re.search(r'S(\d+)G', img_path.split('/')[-1])
                pid = int(match_ds.group(1) + match_sg.group(1))

            camid = self.cam_index

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, pid, camid))
            self.cam_index = self.cam_index + 1

        return data
