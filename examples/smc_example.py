import numpy as np
from samplers.smc import SMC, Proposal_Base
from scipy.stats import norm

def logp(params, data):

    v = 0.1
    mu = 2
    return -0.5 * np.log(v) - 1 / (2 * v) * (params - mu)**2


# Initial proposal
q0 = norm(loc=0, scale=3)

# General proposal
class Proposal(Proposal_Base):

    def __init__(self):
        self.std = 0.1

    def rvs(self, x_cond):
        x_new = x_cond + self.std * np.random.randn()
        return x_new

    def logpdf(self, x, x_cond):
        return -0.5 * np.log(self.std**2) - 1 / (2 * self.std**2) * (x - x_cond)**2

q = Proposal()
smc = SMC(log_target=logp, proposal0=q0, proposal=q)
smc.generate_samples(n_samples=100, data=None, K=100)
