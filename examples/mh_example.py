import numpy as np
from samplers.mh import MH
from matplotlib import pyplot as plt

def logp(params, data):

    v = 0.1
    mu = 2
    return -0.5 * np.log(v) - 1 / (2 * v) * (params[0] - mu)**2

s = MH(log_target=logp)
all_samples, ar = s.generate_samples(all_params_current=[[0], [1]], data=None, proposal_width=0.1, n_samples=5000, n_chains=2, plot_live=False)

fig, ax = plt.subplots()
for samples in all_samples:
    ax.plot(samples[:, 0])

plt.show()
