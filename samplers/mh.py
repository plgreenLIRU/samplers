import numpy as np
from matplotlib import pyplot as plt

class MH:
    def __init__(self, log_target):
        self.log_target = log_target

    def proposal(self, params, proposal_width):
        params_new = params + proposal_width * np.random.randn(len(params))
        return params_new

    def generate_samples(self, params0, data, proposal_width, n_samples, plot_live=False):
        """
        """

        # Initialise algorithm
        params_current = np.array(params0, dtype=float)
        n_params = len(params_current)
        samples = np.zeros((n_samples, n_params), dtype=float)
        proposal_width = np.array(proposal_width, dtype=float)
        logp_current = self.log_target(params=params_current, data=data)
        n_accepted = 0

        # Initialise plotting
        if plot_live:
            plt.ion()
            fig, axes = plt.subplots(n_params, 1, sharex=True, figsize=(8, 2 * n_params))
            if n_params == 1:
                axes = [axes]
            lines = []
            for ax in axes:
                line, = ax.plot([], [], lw=1)
                lines.append(line)
            plt.show()

        # MCMC loop
        i = 0
        while i < n_samples:

            # Proposal
            params_prop = self.proposal(params_current, proposal_width)
            logp_prop = self.log_target(params_prop, data=data)

            # Accept reject
            log_alpha = logp_prop - logp_current
            if np.log(np.random.rand()) < log_alpha:
                params_current = params_prop
                logp_current = logp_prop
                n_accepted += 1

            samples[i] = params_current

            # Plot option
            if plot_live and i % 50 == 0:
                for j in range(n_params):
                    lines[j].set_data(np.arange(i + 1), samples[:i + 1, j])
                    axes[j].relim()
                    axes[j].autoscale_view()
                plt.pause(0.001)

            i += 1

        # Close plot
        if plot_live:
            plt.ioff()
            plt.show()

        final_acc_ratio = 100 * n_accepted / n_samples
        return samples[:i], final_acc_ratio

    def post_process(self, samples, param_names, burn_in=0):
        """
        Plot MCMC results with histograms (top row) and trace plots (bottom row),
        showing burn-in samples in a different colour on the traces.

        Parameters
        ----------
        samples : np.ndarray
            MCMC samples, shape (n_samples, n_params)
        param_names : list of str
            Names of the parameters
        burn_in : int
            Number of initial samples considered burn-in
        """
        n_samples, n_params = samples.shape
        samples_post = samples[burn_in:]

        fig, axes = plt.subplots(2, n_params)

        for i in range(n_params):
            # ------------------
            # Histogram (top row)
            # ------------------
            axes[0, i].hist(samples_post[:, i], bins=30, color='skyblue', edgecolor='k', density=True)
            axes[0, i].set_xlabel(param_names[i])
            axes[0, i].set_ylabel('Density')

            # ------------------
            # Trace plot (bottom row)
            # ------------------
            # burn-in in red
            if burn_in > 0:
                axes[1, i].plot(np.arange(burn_in), samples[:burn_in, i], color='red', label='burn-in')
            # post burn-in in blue
            axes[1, i].plot(np.arange(burn_in, n_samples), samples[burn_in:, i], color='steelblue', label='post burn-in')
            axes[1, i].set_xlabel('Iteration')
            axes[1, i].set_ylabel(param_names[i])
            axes[1, i].legend()

        plt.tight_layout()

        return samples_post
