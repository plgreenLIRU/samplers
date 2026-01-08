import numpy as np
from samplers.smc import SMC, Proposal_Base
from scipy.stats import norm

def logp(params, data):

    v = 0.1
    mu = 2
    return -0.5 * np.log(v) - 1 / (2 * v) * (params - mu)**2

q0 = norm(loc=0, scale=3)

smc = SMC(log_target=logp, proposal0=q0)
smc.generate_samples(n_samples=1000, data=None)
