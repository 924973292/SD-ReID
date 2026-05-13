# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import glob
import re
import mat4py
import pandas as pd
import torch

import os.path as osp

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['AG_ReID_v2', ]


def _resolve_data_dir(root, dataset_dir):
    root = osp.abspath(osp.expanduser(root))
    candidates = [osp.join(root, dataset_dir), osp.join(root, dataset_dir.lower()), root]
    for path in candidates:
        if osp.isdir(path):
            return path
    return candidates[0]


@DATASET_REGISTRY.register()
class AG_ReID_v2(ImageDataset):
    dataset_dir = "AG_ReID_v2"
    dataset_name = 'agreidv2'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.view_dict = {0: 0, 3: 1, 2: 2}  # 0:UAV,3:CCTV, 2:Wearable

        self.root = root
        self.dataset_dir = _resolve_data_dir(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')

        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)

        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        super(AG_ReID_v2, self).__init__(train, query, gallery, **kwargs)

        if verbose:
            print("=> AG-ReID loaded")

            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)
        super().__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)

            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            viewid = self.view_dict[camid]
            camid = self.view_dict[camid]

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            # data.append((img_path, pid, camid, p_attribute))
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        # mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(
            'int64')

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        # mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype('int64')

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3
            # attr = torch.tensor(row[0:].astype(int)) * 2 - 3
            # dict_attribute[str(int(mat_attribute.index[i]))] = self.get_token(attr).to(torch.int64)

        return dict_attribute

    def name_of_attribute(self):
        if not self.key_attribute:
            raise ValueError("Attribute names are not loaded.")
        return self.key_attribute

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


@DATASET_REGISTRY.register()
class AG_ReID_v2_AC(ImageDataset):
    dataset_dir = "AG_ReID_v2"
    dataset_name = 'agreidv2_ag'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.view_dict = {0: 0, 3: 1, 2: 2}  # 0:UAV,3:CCTV, 2:Wearable

        self.root = root
        self.dataset_dir = _resolve_data_dir(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')

        self.q_g_dir = osp.join(self.dataset_dir, 'exp1_aerial_to_cctv.txt')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self._process_dir2(self.q_g_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)

            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            viewid = self.view_dict[camid]
            camid = self.view_dict[camid]

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            # data.append((img_path, pid, camid, p_attribute))
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def _process_dir2(self, file_path, is_train=True):
        with open(file_path) as f:
            img_paths = f.readlines()

        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data_q = []
        data_g = []

        for img_path_row in img_paths:
            if img_path_row.startswith('query'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_q.append((img_path, pid, camid, viewid, p_attribute))

            elif img_path_row.startswith('gallery'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_g.append((img_path, pid, camid, viewid, p_attribute))

        return data_q, data_g

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        # mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(
            'int64')

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        # mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype('int64')

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3
            # attr = torch.tensor(row[0:].astype(int)) * 2 - 3
            # dict_attribute[str(int(mat_attribute.index[i]))] = self.get_token(attr).to(torch.int64)

        return dict_attribute

    def name_of_attribute(self):
        if not self.key_attribute:
            raise ValueError("Attribute names are not loaded.")
        return self.key_attribute

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


@DATASET_REGISTRY.register()
class AG_ReID_v2_AW(ImageDataset):
    dataset_dir = "AG_ReID_v2"
    dataset_name = 'agreidv2_aw'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.view_dict = {0: 0, 3: 1, 2: 2}  # 0:UAV,3:CCTV, 2:Wearable

        self.root = root
        self.dataset_dir = _resolve_data_dir(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')

        self.q_g_dir = osp.join(self.dataset_dir, 'exp2_aerial_to_wearable.txt')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self._process_dir2(self.q_g_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)

            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            viewid = self.view_dict[camid]
            camid = self.view_dict[camid]

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            # data.append((img_path, pid, camid, p_attribute))
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def _process_dir2(self, file_path, is_train=True):
        with open(file_path) as f:
            img_paths = f.readlines()

        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data_q = []
        data_g = []

        for img_path_row in img_paths:
            if img_path_row.startswith('query'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_q.append((img_path, pid, camid, viewid, p_attribute))

            elif img_path_row.startswith('gallery'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_g.append((img_path, pid, camid, viewid, p_attribute))

        return data_q, data_g

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        # mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(
            'int64')

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        # mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype('int64')

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3
            # attr = torch.tensor(row[0:].astype(int)) * 2 - 3
            # dict_attribute[str(int(mat_attribute.index[i]))] = self.get_token(attr).to(torch.int64)

        return dict_attribute

    def name_of_attribute(self):
        if not self.key_attribute:
            raise ValueError("Attribute names are not loaded.")
        return self.key_attribute

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


@DATASET_REGISTRY.register()
class AG_ReID_v2_CA(ImageDataset):
    dataset_dir = "AG_ReID_v2"
    dataset_name = 'agreidv2_ga'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.view_dict = {0: 0, 3: 1, 2: 2}  # 0:UAV,3:CCTV, 2:Wearable

        self.root = root
        self.dataset_dir = _resolve_data_dir(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')

        self.q_g_dir = osp.join(self.dataset_dir, 'exp4_cctv_to_aerial.txt')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self._process_dir2(self.q_g_dir, is_train=False)
        super().__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)

            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            viewid = self.view_dict[camid]
            camid = self.view_dict[camid]

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            # data.append((img_path, pid, camid, p_attribute))
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def _process_dir2(self, file_path, is_train=True):
        with open(file_path) as f:
            img_paths = f.readlines()

        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data_q = []
        data_g = []

        for img_path_row in img_paths:
            if img_path_row.startswith('query'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_q.append((img_path, pid, camid, viewid, p_attribute))


            elif img_path_row.startswith('gallery'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_g.append((img_path, pid, camid, viewid, p_attribute))

        return data_q, data_g

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        # mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(
            'int64')

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        # mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype('int64')

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3
            # attr = torch.tensor(row[0:].astype(int)) * 2 - 3
            # dict_attribute[str(int(mat_attribute.index[i]))] = self.get_token(attr).to(torch.int64)

        return dict_attribute

    def name_of_attribute(self):
        if not self.key_attribute:
            raise ValueError("Attribute names are not loaded.")
        return self.key_attribute

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


@DATASET_REGISTRY.register()
class AG_ReID_v2_WA(ImageDataset):
    dataset_dir = "AG_ReID_v2"
    dataset_name = 'agreidv2_wa'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.view_dict = {0: 0, 3: 1, 2: 2}  # 0:UAV,3:CCTV, 2:Wearable

        self.root = root
        self.dataset_dir = _resolve_data_dir(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')

        self.q_g_dir = osp.join(self.dataset_dir, 'exp5_wearable_to_aerial.txt')

        self.qut_attribute_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self._process_dir2(self.q_g_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)

            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            viewid = self.view_dict[camid]
            camid = self.view_dict[camid]

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            # data.append((img_path, pid, camid, p_attribute))
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def _process_dir2(self, file_path, is_train=True):
        with open(file_path) as f:
            img_paths = f.readlines()

        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data_q = []
        data_g = []

        for img_path_row in img_paths:
            if img_path_row.startswith('query'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]
                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_q.append((img_path, pid, camid, viewid, p_attribute))

            elif img_path_row.startswith('gallery'):
                img_path = osp.join(self.dataset_dir, img_path_row[:-1])

                fname = osp.split(img_path)[-1]

                pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
                pid = int(pid_part1 + pid_part2 + pid_part3)

                camid, frameid = pattern_camid.search(fname).groups()
                camid = int(camid)

                p_attribute = self.attribute_dict_all[str(pid)]
                p_attribute = p_attribute.float()

                viewid = self.view_dict[camid]
                camid = self.view_dict[camid]

                if is_train:
                    pid = "ag-reid" + "_" + str(pid)
                data_g.append((img_path, pid, camid, viewid, p_attribute))

        return data_q, data_g

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        # mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(
            'int64')

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        # mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype('int64')

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3
            # attr = torch.tensor(row[0:].astype(int)) * 2 - 3
            # dict_attribute[str(int(mat_attribute.index[i]))] = self.get_token(attr).to(torch.int64)

        return dict_attribute

    def name_of_attribute(self):
        if not self.key_attribute:
            raise ValueError("Attribute names are not loaded.")
        return self.key_attribute

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
