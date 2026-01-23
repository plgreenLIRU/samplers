import numpy as np
from matplotlib import pyplot as plt

class MH:
    def __init__(self, log_target):
        # Initialize the Metropolis-Hastings sampler with a log-target distribution
        self.log_target = log_target

    def proposal(self, params, proposal_width):
        # Generate a new proposal by adding Gaussian noise to current parameters
        params_new = params + proposal_width * np.random.randn(len(params))
        return params_new

    def generate_samples(self, params_current, data, proposal_width, n_samples, n_chains=1, plot_live=False):
        # Run Metropolis-Hastings MCMC to generate posterior samples

        # Initialise algorithm
        n_params = len(params_current[0])

        samples = []
        logp_current = []
        n_accepted = []
        for params in params_current:
            samples.append(np.zeros((n_samples, n_params), dtype=float))
            logp_current.append(self.log_target(params=params, data=data))
            n_accepted.append(0)

        proposal_width = np.array(proposal_width, dtype=float)  # can remove?

        # Initialise plotting
        if plot_live:
            plt.ion()
            fig, axes = plt.subplots(n_params, 1, sharex=True, figsize=(8, 2 * n_params))
            if n_params == 1:
                axes = [axes]
            all_lines = []
            for _ in range(n_chains):
                lines = []
                for ax in axes:
                    line, = ax.plot([], [], lw=1)
                    lines.append(line)
                all_lines.append(lines)
            plt.show()

        # MCMC loop
        i = 0
        while i < n_samples:

            for chain in range(n_chains):

                # Proposal
                params_prop = self.proposal(params_current[chain], proposal_width)
                logp_prop = self.log_target(params_prop, data=data)

                # Accept reject
                log_alpha = logp_prop - logp_current[chain]
                if np.log(np.random.rand()) < log_alpha:
                    params_current[chain] = params_prop
                    logp_current[chain] = logp_prop
                    n_accepted[chain] += 1

                samples[chain][i] = params_current[chain]

            # Plot option
            if plot_live and i % 50 == 0:
                for chain in range(n_chains):
                    for j in range(n_params):
                        all_lines[chain][j].set_data(np.arange(i + 1), samples[chain][:i + 1, j])
                        axes[j].relim()
                        axes[j].autoscale_view()
                plt.pause(0.001)

            i += 1

        # Close plot
        if plot_live:
            plt.ioff()
            plt.show()

        final_acc_ratio = []
        for n in n_accepted:
            final_acc_ratio.append(100 * n / n_samples)
        return samples, final_acc_ratio

    def post_process(self, samples, param_names, burn_in=0):
        # Visualize posterior samples with histograms after removing burn-in period

        n_chains = len(samples)
        n_samples, n_params = samples[0].shape
        results_figures = []
        all_samples = []
   
        for i in range(n_params):
            fig, ax = plt.subplots()
            s = np.array([])
            for chain in range(n_chains):
                s = np.append(s, samples[chain][burn_in:, i])

            all_samples.append(s)

            # Histogram (top row)
            ax.hist(s, bins=30, color='skyblue', edgecolor='k', density=True)
            ax.set_xlabel(param_names[i])
            ax.set_ylabel('Density')
            results_figures.append(fig)

        return results_figures, np.array(all_samples)
