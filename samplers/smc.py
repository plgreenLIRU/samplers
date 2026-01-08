import numpy as np

class Proposal_Base:
    def __init__(self):
        pass

    def rvs(self, x_cond):
        pass

    def logpdf(self, x, x_cond):
        pass

class SMC:
    def __init__(self, log_target, proposal0, proposal):
        self.log_target = log_target
        self.proposal0 = proposal0
        self.proposal = proposal

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
        resampled_weights = np.repeat(1/len(samples), len(samples))
        return resampled_samples, resampled_weights

    def generate_samples(self, data, n_samples, K):
        import numpy as np
        import matplotlib.pyplot as plt

        # ----------------------------
        # Live plotting setup
        # ----------------------------
        plt.ion()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        line_mean, = ax1.plot([], [], lw=2)
        ax1.set_title("Running Mean")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Mean")

        line_ess, = ax2.plot([], [], lw=2)
        ax2.set_title("Running ESS")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("ESS")

        # ----------------------------
        # Initialise estimates
        # ----------------------------
        running_mean = []
        running_ess = []

        # Initial samples
        X = self.proposal0.rvs(n_samples)
        log_weights = []
        for x in X:
            log_weights.append(self.log_target(x, data) - self.proposal0.pdf(x))
        log_weights = np.array(log_weights)

        # ----------------------------
        # Main loop
        # ----------------------------
        for k in range(K):

            # Find normalised weights
            weights = self.find_normalised_weights(log_weights)

            # Compute estimates
            mean, cov = self.find_mean_cov(X, weights)
            running_mean.append(mean)

            ess = self.find_ESS(weights)
            running_ess.append(ess)

            if ess < 50:
                print('resampled')
                X, weights = self.resample(X, weights)
                log_weights = np.log(weights)

            # ----------------------------
            # Update plots
            # ----------------------------
            xvals = range(len(running_mean))

            line_mean.set_data(xvals, running_mean)
            ax1.relim()
            ax1.autoscale_view()

            line_ess.set_data(xvals, running_ess)
            ax2.relim()
            ax2.autoscale_view()

            plt.tight_layout()
            plt.pause(0.01)

            # ----------------------------
            # Propose new
            # ----------------------------
            X_new = []
            for x in X:
                x_new = self.proposal.rvs(x_cond=x)
                X_new.append(x_new)
            X_new = np.array(X_new)

            # Compute new weights
            log_weights_new = []
            for x, x_new, logw in zip(X, X_new, log_weights):
                log_weights_new.append(
                    self.log_target(x_new, data=data)
                    - self.log_target(x, data=data)
                    + logw
                )

            # Update
            log_weights = np.array(log_weights_new)
            X = np.array(X_new)

        # ----------------------------
        # Finalize plot
        # ----------------------------
        plt.ioff()
        plt.show()

        return X, log_weights

