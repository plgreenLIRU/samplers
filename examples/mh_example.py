import numpy as np
from samplers.mh import MH
from matplotlib import pyplot as plt

def logp(params, data):

    v = 0.1
    mu = 2
    return -0.5 * np.log(v) - 1 / (2 * v) * (params[0] - mu)**2

mh = MH(log_target=logp)
samples, ar = mh.generate_samples(params_current=[[0], [1], [2], [3]],
                                  data=None, proposal_width=0.1, n_samples=5000,
                                  n_chains=4, plot_live=True)

mh.post_process(samples=samples, param_names=['x'], burn_in=1000)
