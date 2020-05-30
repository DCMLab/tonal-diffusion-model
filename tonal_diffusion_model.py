import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric


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
                 margin=0.5,
                 min_iterations=None,
                 max_iterations=1000):
        super().__init__()
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)
        self.margin = margin
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.effective_min_iterations = None
        self.log_interval_step_weights = Parameter(torch.tensor([]))
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.reference_center = None
        self.interval_class_distribution = None
        self.init_interval_class_distribution = None
        self.transition_matrix = None
        self.data = None
        self.matched_dist = None
        self.matched_loss = None
        self.matched_shift = None
        self.iteration = 0

    def set_data(self, pitch_class_distributions):
        self.iteration = 0
        self.data = torch.from_numpy(pitch_class_distributions)
        self.n_data, self.n_interval_classes = pitch_class_distributions.shape
        # get necessary support of distribution, reference center, and minimum number of iterations to reach every point
        self.n_dist_support = int(np.ceil((2 * self.margin + 1) * self.n_interval_classes))
        self.reference_center = int(np.round((self.margin + 0.5) * self.n_interval_classes))
        if self.min_iterations is None:
            largest_step_down = -min(np.min(self.interval_steps), 0)
            largest_step_up = max(np.max(self.interval_steps), 0)
            self.effective_min_iterations = int(np.ceil(max(
                self.reference_center / largest_step_down,
                (self.n_dist_support - self.reference_center) / largest_step_up
            ))) + 1
        else:
            self.effective_min_iterations = self.min_iterations
        # initialise interval class distributions with single tonal center
        self.init_interval_class_distribution = np.zeros((self.n_data, self.n_dist_support))
        self.init_interval_class_distribution[:, self.reference_center] = 1
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
        # initialise weights
        self.log_interval_step_weights.data = torch.zeros((self.n_data, self.n_interval_steps), dtype=torch.float64)

    def set_params(self, log_weights):
        assert tuple(log_weights.shape) == (self.n_data, self.n_interval_steps)
        self.log_interval_step_weights.data = torch.from_numpy(log_weights)
        self.zero_grad()

    def get_params(self):
        return self.log_interval_step_weights.data.numpy().copy()

    def get_interpretable_params(self):
        weight_sum = self.log_interval_step_weights.exp().sum(dim=1).data.numpy()
        weights = self.log_interval_step_weights.exp().data.numpy() / weight_sum[:, None]
        params = np.zeros((weights.shape[0], weights.shape[1] + 1))
        params[:, 0] = weight_sum
        params[:, 1:] = weights
        return params

    def perform_diffusion(self):
        # create path length distributions and cumulative sum to track convergence
        weight_sum = self.log_interval_step_weights.exp().sum(dim=1)
        path_length_dist = Poisson(rate=weight_sum)
        # path_length_dist = Geometric(probs=weight_sum.log().sigmoid())
        cum_path_length_prob = torch.zeros_like(weight_sum)
        # get interval step probabilities
        interval_step_probs = self.log_interval_step_weights.exp() / weight_sum[:, None]
        # initialise running and output distributions
        running_interval_class_distribution = self.init_interval_class_distribution
        self.interval_class_distribution = torch.zeros_like(self.init_interval_class_distribution)
        # marginalise latent variable
        for n in range(self.max_iterations):
            if np.allclose(cum_path_length_prob.data.numpy(), 1) and n >= self.effective_min_iterations:
                break
            # probability to reach this step
            step_prob = path_length_dist.log_prob(n).exp()
            # update output distributions (marginalise path length)
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
        else:
            raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")
        # normalise to account for border effects and path length cut-off
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)
        if np.any(np.isnan(self.interval_class_distribution.data.numpy())):
            print(self.interval_class_distribution)
            raise RuntimeWarning("got nan")

    @staticmethod
    def dkl(p, q, dim=None):
        zeros = torch.zeros_like(p)
        dkl = torch.where(torch.isclose(p, zeros),
                          zeros,
                          p * (p.log() - q.log())).sum(dim=dim)
        if not np.all(torch.isfinite(dkl).data.numpy()):
            print(dkl)
            raise RuntimeWarning("Got non-finite (inf/nan)")
        return dkl

    def match_distributions(self):
        n_shifts = self.n_dist_support - self.n_interval_classes + 1
        self.matched_dist = None
        self.matched_loss = None
        self.matched_shift = np.zeros(self.n_data, dtype=int)
        for shift in range(n_shifts):
            dist = self.interval_class_distribution[:, shift:shift + self.n_interval_classes]
            dist = dist / dist.sum(dim=1, keepdim=True)
            loss = self.dkl(self.data, dist, dim=1)
            if shift == 0:
                self.matched_dist = dist
                self.matched_loss = loss
            else:
                cond = loss < self.matched_loss
                self.matched_dist = torch.where(cond[:, None], dist, self.matched_dist)
                self.matched_loss = torch.where(cond, loss, self.matched_loss)
                self.matched_shift = np.where(cond, shift, self.matched_shift)

    def get_distributions(self):
        self.perform_diffusion()
        self.match_distributions()
        return self.matched_dist.data.numpy().copy()

    def get_loss(self):
        return self.matched_loss.data.numpy().copy()

    def get_centers(self):
        return self.reference_center - self.matched_shift

    def loss(self, log_weights, grad=False):
        self.set_params(log_weights=log_weights.reshape((self.n_data, self.n_interval_steps)))
        self.perform_diffusion()
        self.match_distributions()
        if grad:
            loss = self.matched_loss.mean()
            loss.backward()
            _grad = self.log_interval_step_weights.grad.data.numpy()
            if np.any(np.isnan(_grad)):
                print(_grad)
                raise RuntimeWarning("got nan")
            return loss.data.numpy(), _grad
        else:
            with torch.no_grad():
                loss = self.matched_loss.mean()
                return loss.data.numpy()

    def grad(self, log_weights):
        loss, grad = self.loss(log_weights=log_weights, grad=True)
        return grad.flatten()

    def closure(self):
        self.zero_grad()
        self.perform_diffusion()
        self.match_distributions()
        loss = self.matched_loss.mean()
        loss.backward()
        return loss

    def callback(self, log_weights):
        self.iteration += 1
        print(f"iteration {self.iteration}")
        print(f"    loss: {self.loss(log_weights=log_weights)}")


class StaticDistributionModel(Module):

    def __init__(self,
                 margin=0.5,
                 max_iterations=1000):
        super().__init__()
        self.margin = margin
        self.max_iterations = max_iterations
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.reference_center = None
        self.static_interval_log_distribution = Parameter(torch.tensor([]))
        self.interval_class_distribution = None
        self.init_interval_class_distribution = None
        self.transition_matrix = None
        self.log_data = None
        self.matched_log_dist = None
        self.matched_loss = None
        self.matched_shift = None
        self.iteration = 0

    def set_data(self, pitch_class_distributions):
        self.iteration = 0
        self.log_data = torch.from_numpy(np.log(pitch_class_distributions))
        self.n_data, self.n_interval_classes = pitch_class_distributions.shape
        # get necessary support of distribution, reference center, and minimum number of iterations to reach every point
        self.n_dist_support = int(np.ceil((2 * self.margin + 1) * self.n_interval_classes))
        self.reference_center = int(np.round((self.margin + 0.5) * self.n_interval_classes))
        # initialise distribution with mode at reference center
        self.static_interval_log_distribution.data = torch.from_numpy(
            -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 5) ** 2
        )

    def set_params(self, log_dist):
        assert tuple(log_dist.shape) == (self.n_dist_support,)
        self.static_interval_log_distribution.data = torch.from_numpy(log_dist)
        self.zero_grad()

    def get_params(self):
        return self.static_interval_log_distribution.data.numpy().copy()

    def get_interpretable_params(self):
        return [[] for _ in range(self.n_data)]

    @staticmethod
    def dkl_log(log_p, log_q, dim=None):
        zeros = torch.zeros_like(log_p)
        dkl = torch.where(torch.isfinite(log_p),
                          log_p.exp() * (log_p - log_q),
                          zeros).sum(dim=dim)
        if not np.all(torch.isfinite(dkl).data.numpy()):
            print(dkl)
            raise RuntimeWarning("Got non-finite (inf/nan)")
        return dkl

    def match_distributions(self):
        n_shifts = self.n_dist_support - self.n_interval_classes + 1
        self.matched_log_dist = None
        self.matched_loss = None
        self.matched_shift = np.zeros(self.n_data, dtype=int)
        for shift in range(n_shifts):
            log_dist = self.static_interval_log_distribution[None, shift:shift + self.n_interval_classes]
            log_dist = log_dist - log_dist.logsumexp(dim=1, keepdim=True)
            loss = self.dkl_log(self.log_data, log_dist, dim=1)
            if shift == 0:
                self.matched_log_dist = log_dist
                self.matched_loss = loss
            else:
                cond = loss < self.matched_loss
                self.matched_log_dist = torch.where(cond[:, None], log_dist, self.matched_log_dist)
                self.matched_loss = torch.where(cond, loss, self.matched_loss)
                self.matched_shift = np.where(cond, shift, self.matched_shift)

    def get_distributions(self):
        self.match_distributions()
        return self.matched_log_dist.exp().data.numpy().copy()

    def get_loss(self):
        return self.matched_loss.data.numpy().copy()

    def get_centers(self):
        return self.reference_center - self.matched_shift

    def loss(self, log_dist, grad=False):
        self.set_params(log_dist=log_dist)
        self.match_distributions()
        if grad:
            loss = self.matched_loss.mean()
            loss.backward()
            _grad = self.static_interval_log_distribution.grad.data.numpy()
            if np.any(np.isnan(_grad)):
                print(_grad)
                raise RuntimeWarning("got nan")
            return loss.data.numpy(), _grad
        else:
            with torch.no_grad():
                loss = self.matched_loss.mean()
                return loss.data.numpy()

    def grad(self, log_dist):
        loss, grad = self.loss(log_dist=log_dist, grad=True)
        return grad.flatten()

    def closure(self):
        self.zero_grad()
        self.match_distributions()
        loss = self.matched_loss.mean()
        loss.backward()
        return loss

    def callback(self, log_dist):
        self.iteration += 1
        print(f"iteration {self.iteration}")
        print(f"    loss: {self.loss(log_dist=log_dist)}")


class GaussianModel:

    def __init__(self):
        self.data = None
        self.mean = None
        self.var = None
        self.distributions = None
        self.loss = None

    def set_data(self, pitch_class_distributions):
        self.data = pitch_class_distributions
        pos = np.arange(pitch_class_distributions.shape[1])
        self.mean = (pitch_class_distributions * pos).sum(axis=1)
        self.var = (pitch_class_distributions * (pos[None, :] - self.mean[:, None]) ** 2).sum(axis=1)
        self.distributions = np.exp(-(pos[None, :] - self.mean[:, None]) ** 2 / self.var[:, None] / 2)
        self.distributions /= self.distributions.sum(axis=1, keepdims=True)
        self.loss = np.where(self.data == 0,
                             np.zeros_like(self.data),
                             self.data * (np.log(self.data) - np.log(self.distributions))).sum(axis=1)

    def get_distributions(self):
        return self.distributions.copy()

    def get_loss(self):
        return self.loss.copy()

    def get_centers(self):
        return self.mean.copy()

    def get_interpretable_params(self):
        return np.sqrt(self.var)[:, None]
