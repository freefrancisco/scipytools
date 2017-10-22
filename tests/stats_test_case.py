import unittest
import numpy as np
import scipy as sp
from scipy.stats import uniform
from stats import FuncDistCompose

class StatsTestCase(unittest.TestCase):
    def setUp(self):
        self.funcdist = FuncDistCompose(np.exp, sp.stats.uniform)
        self.funcdist(-4*np.log(10), 8*np.log(10))


    def test_rvs(self):
        nums = self.funcdist.rvs(10000)
        print(np.min(nums), np.max(nums))
        self.assertAlmostEqual(np.min(nums), 1e-4, delta=1e-6)
        self.assertAlmostEqual(np.max(nums), 1e4, delta=100)
        self.assertEqual(len(nums), 10000)


if __name__ == '__main__':
    unittest.main()
