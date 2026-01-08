import numpy as np

class Proposal_Base:
    def __init__(self):
        pass

    def rvs(self, Ns, x_cond=None):
        pass

    def logpdf(self, x_cond=None):
        pass

class SMC:
    def __init__(self, log_target, proposal0):
        self.log_target = log_target
        self.proposal0 = proposal0

    def find_normalised_weights(self, log_weights):

        max_log_w = np.max(log_weights)
        stable_weights = np.exp(log_weights - max_log_w)
        normalised_weights = stable_weights / np.sum(stable_weights)
        return normalised_weights

    def find_mean_cov(self, X, weights):
                
        # Weighted mean
        mean = np.sum(weights * X)
        
        # Weighted covariance
        X_centered = X - mean
        cov = (weights * X_centered).T @ X_centered
        
        return mean, cov

    def find_ESS(self, w):
        return 1 / np.sum(w**2) / len(w) * 100

    def resample(self, samples, weights):

        indices = np.random.choice(len(samples), size=len(weights), replace=True, p=weights)
        resampled_samples = samples[indices]
        return resampled_samples

    def generate_samples(self, data, n_samples):
        
        # Initialise estimates
        estimated_mean = []
        estimated_cov = []

        # Initial samples
        X = self.proposal0.rvs(n_samples)
        log_weights = []
        for x in X:
            log_weights.append(self.log_target(x, data) - self.proposal0.pdf(x))

        # Convert to array
        log_weights = np.array(log_weights)

        # Find normalised
        w = self.find_normalised_weights(log_weights)

        # Compute estimates
        mean, cov = self.find_mean_cov(X, w)
        
        ess = self.find_ESS(w)

        print(mean)
        print(cov)
        print(ess)

        if ess < 50:
            print('resampled')
            X = self.resample(X, w)
