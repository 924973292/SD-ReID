import unittest
import sys
from itertools import islice

sys.path.append('.')
from fastreid.data.samplers import TrainingSampler


class SamplerTestCase(unittest.TestCase):
    def test_training_sampler(self):
        sampler = TrainingSampler(5, shuffle=False, seed=1)
        self.assertEqual(list(islice(sampler, 5)), [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
