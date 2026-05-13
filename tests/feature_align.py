import unittest
import numpy as np
import os


class TestFeatureAlign(unittest.TestCase):
    def test_caffe_pytorch_feat_align(self):
        caffe_feat_path = os.getenv("FASTREID_CAFFE_FEAT_DIR")
        pytorch_feat_path = os.getenv("FASTREID_PYTORCH_FEAT_DIR")
        if not caffe_feat_path or not pytorch_feat_path:
            self.skipTest("set FASTREID_CAFFE_FEAT_DIR and FASTREID_PYTORCH_FEAT_DIR to run this alignment test")
        feat_filenames = os.listdir(caffe_feat_path)
        for feat_name in feat_filenames:
            caffe_feat = np.load(os.path.join(caffe_feat_path, feat_name))
            pytorch_feat = np.load(os.path.join(pytorch_feat_path, feat_name))
            sim = np.dot(caffe_feat, pytorch_feat.transpose())[0][0]
            assert sim > 0.97, f"Got similarity {sim} and feature of {feat_name} is not aligned"

    def test_model_performance(self):
        caffe_feat_path = os.getenv("FASTREID_CAFFE_FEAT_DIR")
        if not caffe_feat_path:
            self.skipTest("set FASTREID_CAFFE_FEAT_DIR to run this performance smoke test")
        feat_filenames = os.listdir(caffe_feat_path)
        feats = []
        for feat_name in feat_filenames:
            caffe_feat = np.load(os.path.join(caffe_feat_path, feat_name))
            feats.append(caffe_feat)
        self.assertGreater(len(feats), 0)



if __name__ == '__main__':
    unittest.main()
