#
#  Author Francisco Gutierrez 2017
#

# this is intended to compose a function with a distribution from scipy.stats,
# for now it is only a hack to make it work with the randomized search for hyperparameters in scikit-learn
# sklearn.model_selection.RandomizedSearchCV requires a distribution with a method rvs that it calls to generate random numbers of that distribution.
# in this particular case I wanted a continuous sample of a variable that came from a np.logspace, so a variable that is
# uniform in log space, so basically what I needed was a composition of np.exp with the distribution scipy.stats.uniform
# this seems to work with the randomized search.
#

import numpy as np
import scipy as sp

class FuncDistCompose(object):
    def __init__(self, func, dist):
        self.func = func
        self.dist = dist

    def __call__(self, *args, **kwargs):
        self.dist = self.dist(*args, **kwargs)
        return self

    def rvs(self, *args, **kwargs):
        return self.func(self.dist.rvs(*args, **kwargs))
