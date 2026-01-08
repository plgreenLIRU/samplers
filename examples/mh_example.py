import numpy as np
from samplers.mh import MH

def logp(params, data):

    v = 0.1
    mu = 2
    return -0.5 * np.log(v) - 1 / (2 * v) * (params - mu)**2

s = MH(log_target=logp)
samples, ar = s.generate_samples(params0=np.array([0.]), data=None, proposal_width=0.1, n_samples=5000, plot_live=True)
