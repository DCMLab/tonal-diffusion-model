import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions.poisson import Poisson

import matplotlib.pyplot as plt

from util import NestedOutputSingleton, NestedOutputDummy, normalize_non_zero
NO = NestedOutputSingleton
NONO = NestedOutputDummy


class TonalDiffusionModel(Module):

    def __init__(self,
                 interval_steps=(
                         1,   # fifth up
                         -1,  # fifth down
                         -3,  # minor third up
                         3,   # minor third down
                         4,   # major third up
                         -4   # major third down
                 ),
                 init_lambda=1.,
                 margin=1,
                 max_iterations=1000):
        super().__init__()
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)
        self.init_lambda = init_lambda
        self.margin = margin
        self.max_iterations = max_iterations
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.log_interval_step_weights = Parameter(torch.tensor([]))
        self.interval_class_distribution = None
        self.init_interval_class_distribution = None
        self.transition_matrix = None
        self.data = None

    def set_data(self, pitch_class_distributions):
        self.data = torch.from_numpy(pitch_class_distributions)
        self.n_data, self.n_interval_classes = pitch_class_distributions.shape
        # initialise weights so that they sum up to init_lambda
        self.log_interval_step_weights.data = torch.from_numpy(
            np.zeros((self.n_data, self.n_interval_steps)) + np.log(self.init_lambda) - np.log(self.n_interval_steps)
        )
        # initialise interval class distribution with single tonal center
        self.n_dist_support = 2 * self.margin * self.n_interval_classes + 1
        self.init_interval_class_distribution = np.zeros((self.n_data, self.n_dist_support))
        self.init_interval_class_distribution[:, self.margin * self.n_interval_classes] = 1
        self.init_interval_class_distribution = torch.from_numpy(self.init_interval_class_distribution)
        # init transition matrix
        self.transition_matrix = np.zeros((self.n_dist_support, self.n_dist_support, self.n_interval_steps))
        for interval_index, interval_step in enumerate(self.interval_steps):
            if interval_step > 0:
                from_indices = np.arange(0, self.n_dist_support - interval_step)
            else:
                from_indices = np.arange(-interval_step, self.n_dist_support)
            to_indices = from_indices + interval_step
            self.transition_matrix[from_indices, to_indices, interval_index] = 1
        self.transition_matrix = torch.from_numpy(self.transition_matrix)
        self.transition_matrix.requires_grad = False

    def get_interval_class_distribution(self):
        # create path length distribution and cumulative sum to track convergence
        lam = self.log_interval_step_weights.exp().sum(dim=1)
        path_length_dist = Poisson(rate=lam)
        cum_path_length_prob = torch.zeros_like(lam, requires_grad=False)
        # get interval step probabilities
        interval_step_probs = self.log_interval_step_weights.exp() / lam[:, None]
        # initialise running and output distribution
        running_interval_class_distribution = self.init_interval_class_distribution
        self.interval_class_distribution = torch.zeros_like(self.init_interval_class_distribution)
        # marginalise latent variable
        for n in range(self.max_iterations):
            NO.print(f"iteration {n}")
            with NO():
                if np.allclose(cum_path_length_prob.data.numpy(), 1):
                    break
                # probability to reach this step
                step_prob = path_length_dist.log_prob(n).exp()
                # update output distribution (marginalise path length)
                NO.print(f"self.interval_class_distribution: {self.interval_class_distribution.shape}")
                NO.print(f"running_interval_class_distribution: {running_interval_class_distribution.shape}")
                NO.print(f"step_prob: {step_prob.shape}")
                self.interval_class_distribution = self.interval_class_distribution + \
                                                   step_prob[:, None] * running_interval_class_distribution
                # perform transition (marginalise interval classes)
                # intermediate tensor has dimensions:
                # (n_data, n_dist_support, n_dist_support, n_interval_steps) = (data, from, to, interval)
                running_interval_class_distribution = torch.einsum("fti,di,df->dt",
                                                                   self.transition_matrix,
                                                                   interval_step_probs,
                                                                   running_interval_class_distribution)
                # update cumulative
                cum_path_length_prob = cum_path_length_prob + step_prob
                NO.print(f"cum: {cum_path_length_prob}")
        else:
            raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")
        # normalise to account for border effects and path length cut-off
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)
        return self.interval_class_distribution


if __name__ == "__main__":
    # generate data
    n_data = 10
    n_pitch_classes = 20
    data = np.zeros((n_data, n_pitch_classes))
    for data_idx in range(n_data):
        mean = np.random.randint(0, n_pitch_classes)
        std = np.random.uniform(1, 5)
        data[data_idx] = np.exp(-((np.arange(n_pitch_classes) - mean) / (2 * std)) ** 2)
    normalize_non_zero(data)

    # set up model and get distributions
    model = TonalDiffusionModel()
    model.set_data(data)
    distributions = model.get_interval_class_distribution().data.numpy()

    # set up plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # plot data
    ax = axes[0]
    for d in data:
        ax.plot(d, '-o')

    # plot model
    ax = axes[1]
    for d in distributions:
        ax.plot(d, '-o')

    plt.show()
