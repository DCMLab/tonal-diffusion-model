import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial
from torch.distributions.gamma import Gamma


def fullnp():
    return np.printoptions(threshold=np.inf)


class IntervalClassModel(Module):

    @staticmethod
    def dkl(p, q, dim=None):
        zeros = torch.zeros_like(p)
        dkl = torch.where(torch.isclose(p, zeros),
                          zeros,
                          p * (p.log() - q.log())).sum(dim=dim)
        if np.any(torch.isnan(dkl).data.numpy()):
            print(dkl)
            raise RuntimeWarning("Got nan")
        return dkl

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

    def __init__(self):
        super().__init__()
        self.iteration = 0

    def forward(self, *input):
        raise RuntimeWarning("This is not a normal Module")

    def set_data(self, data, weights=None):
        self.iteration = 0
        self.data = torch.from_numpy(data)
        self.n_data, self.n_interval_classes = data.shape
        # set data weights
        if weights is not None:
            self.data_weights = torch.from_numpy(weights)
            self.data_weights = self.data_weights / self.data_weights.sum()
        else:
            self.data_weights = None

    def get_params(self, **kwargs):
        return np.concatenate(tuple(p.data.numpy().flatten() for p in self.parameters(**kwargs)))

    def set_params(self, params, **kwargs):
        if np.any(np.isnan(params)):
            raise RuntimeWarning(f"nan params: {params}")
        if len(params.shape) > 1:
            # expect all shapes to match
            for p, n in zip(self.parameters(**kwargs), params):
                p.data = torch.from_numpy(n)
        else:
            # reshape consecutive segments
            idx = 0
            for p in self.parameters(**kwargs):
                size = np.prod(p.shape)
                p.data = torch.from_numpy(params[idx:idx + size].reshape(p.shape))
                idx += size

    def _loss(self):
        raise NotImplementedError

    def loss(self, params, **kwargs):
        self.set_params(params, **kwargs)
        self.zero_grad()
        with torch.no_grad():
            return self._loss().data.numpy()

    def grad(self, params, *, return_loss=False, **kwargs):
        self.set_params(params, **kwargs)
        self.zero_grad()
        loss = self._loss()
        loss.backward()
        grad = np.concatenate(tuple(p.grad.data.numpy().flatten() for p in self.parameters(**kwargs)))
        if np.any(np.isnan(grad)):
            with fullnp():
                raise RuntimeWarning(f"grad has nan values: {grad}\nloss: {loss}")
        if return_loss:
            return loss.data.numpy().copy(), grad
        else:
            return grad

    def closure(self):
        self.zero_grad()
        loss = self._loss()
        loss.backward()
        return loss

    def callback(self, params):
        self.iteration += 1
        print(f"iteration {self.iteration}")
        print(f"    loss: {self.loss(params)}")


class TonnetzModel(IntervalClassModel):

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
                 *args,
                 **kwargs):
        super().__init__()
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)
        self.margin = margin
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.reference_center = None
        self.interval_class_distribution = None
        self.data = None
        self.matched_dist = None
        self.matched_loss = None
        self.matched_shift = None
        self.data_weights = None

    def _loss(self):
        self.match_distributions()
        if self.data_weights is None:
            return self.matched_loss.mean()
        else:
            return (self.matched_loss * self.data_weights).sum()

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # get necessary support of distribution and reference center
        self.n_dist_support = int(np.ceil((2 * self.margin + 1) * self.n_interval_classes))
        self.reference_center = int(np.round((self.margin + 0.5) * self.n_interval_classes))

    def get_interpretable_params(self):
        raise NotImplementedError

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

    def get_results(self):
        """
        Compute model loss and return internal results
        :return: tuple of arrays (matched distribution, loss, center) containing results per data point
        """
        # compute everything
        self._loss()
        # return internal results
        return (self.matched_dist.data.numpy().copy(),
                self.matched_loss.data.numpy().copy(),
                self.reference_center - self.matched_shift)


class DiffusionModel(TonnetzModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_interval_class_distribution = None
        self.transition_matrix = None

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
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

    def get_interpretable_params(self):
        raise NotImplementedError

    def perform_diffusion(self):
        raise NotImplementedError

    def _loss(self):
        self.perform_diffusion()
        return super()._loss()


class TonalDiffusionModel(DiffusionModel):

    def __init__(self,
                 min_iterations=None,
                 max_iterations=100,
                 path_dist=Binomial,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.effective_min_iterations = None
        self.log_interval_step_weights = Parameter(torch.tensor([]))
        self.path_log_params = Parameter(torch.tensor([]))
        self.path_dist = path_dist
        if path_dist == Gamma:
            self.precompute_path_dist = True
        else:
            self.precompute_path_dist = False

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # set minimum number of iterations to reach every point
        if self.min_iterations is None:
            largest_step_down = -min(np.min(self.interval_steps), 0)
            largest_step_up = max(np.max(self.interval_steps), 0)
            self.effective_min_iterations = int(np.ceil(max(
                self.reference_center / largest_step_down,
                (self.n_dist_support - self.reference_center) / largest_step_up
            ))) + 1
        else:
            self.effective_min_iterations = self.min_iterations
        # initialise weights
        self.log_interval_step_weights.data = torch.zeros((self.n_data, self.n_interval_steps), dtype=torch.float64)
        # initialise distribution parameters
        if self.path_dist in [Poisson, Geometric]:
            self.path_log_params.data = torch.zeros(self.n_data, dtype=torch.float64)
        elif self.path_dist in [Gamma, Binomial]:
             params = np.ones((self.n_data, 2), dtype=np.float64)
             params[:, 0] *= 2
             params[:, 1] *= -2
             self.path_log_params.data = torch.from_numpy(params)
        else:
            raise RuntimeWarning("Failed Case")

    def get_interpretable_params(self):
        weight_sum = self.log_interval_step_weights.exp().sum(dim=1).data.numpy()
        weights = self.log_interval_step_weights.exp().data.numpy() / weight_sum[:, None]
        # # separate_path_params:
        #     params = np.zeros((weights.shape[0], weights.shape[1] + self.path_log_params.shape[1]))
        #     params[:, :weights.shape[1]] = weights
        #     params[:, -self.path_log_params.shape[1]:] = self.path_log_params.data.numpy()
        params = np.zeros((weights.shape[0], weights.shape[1] + 1))
        params[:, :weights.shape[1]] = weights
        params[:, -1] = weight_sum
        return params

    def perform_diffusion(self):
        # float offset to n (hack e.g. for Gamma, which is not defined for n=0)
        if self.path_dist == Gamma:
            n_offset = 1e-50
        else:
            n_offset = 0
        # uniform offset of probability distribution to ensure finite KL divergence if model produces strict zero
        # probabilities for some pitch classes otherwise (e.g. for Binomial)
        if self.path_dist == Binomial:
            epsilon = 1e-50
        else:
            epsilon = 0
        # path length distribution
        if self.path_dist == Poisson:
            path_length_dist = Poisson(rate=self.path_log_params.exp())
        elif self.path_dist == Geometric:
            path_length_dist = Geometric(probs=self.path_log_params.sigmoid())
        elif self.path_dist == Gamma:
            path_length_dist = Gamma(concentration=self.path_log_params[:, 0].exp(),
                                     rate=self.path_log_params[:, 1].exp())
        elif self.path_dist == Binomial:
            total_count = self.path_log_params[:, 0].exp()
            total_count_floor = total_count.floor()
            total_count_ceil = total_count.ceil()
            alpha = total_count - total_count_floor
            probs = self.path_log_params[:, 1].sigmoid()
            floor_bin = Binomial(total_count=total_count_floor, probs=probs)
            ceil_bin = Binomial(total_count=total_count_ceil, probs=probs)
        else:
            raise RuntimeWarning("Failed Case")
        # callable
        if self.path_dist == Binomial:
            def path_length_dist_func(n):
                return alpha * ceil_bin.log_prob(n).exp() + (1 - alpha) * floor_bin.log_prob(n).exp()
        else:
            def path_length_dist_func(n):
                return path_length_dist.log_prob(n).exp()
        # normalise path length distribution
        if self.path_dist == Gamma:
            l = []
            for n in range(self.max_iterations):
                n = torch.tensor([n + n_offset], dtype=torch.float64)
                l.append(path_length_dist_func(n))
            path_length_dist_arr = torch.stack(l)
            normalisation = path_length_dist_arr.sum(dim=0, keepdim=True)
            assert not np.any(np.isclose(normalisation.data.numpy(), 0)), normalisation.data.numpy().tolist()
            path_length_dist_arr = path_length_dist_arr / normalisation
        # cumulative sum to track convergence
        cum_path_length_prob = torch.zeros(self.n_data, dtype=torch.float64)
        # get interval step probabilities
        interval_step_log_probs = self.log_interval_step_weights - \
                                  self.log_interval_step_weights.logsumexp(dim=1, keepdim=True)
        # initialise running and output distributions
        running_interval_class_distribution = self.init_interval_class_distribution
        self.interval_class_distribution = torch.zeros_like(self.init_interval_class_distribution)
        # marginalise latent variable
        for n in range(self.max_iterations):
            # probability to reach this step
            if self.path_dist == Gamma:
                step_prob = path_length_dist_arr[n]
            else:
                step_prob = path_length_dist_func(torch.tensor(n, dtype=torch.float64))
            # update output distributions (marginalise path length)
            self.interval_class_distribution = self.interval_class_distribution + \
                                               step_prob[:, None] * running_interval_class_distribution
            # perform transition (marginalise interval classes)
            # intermediate tensor has dimensions:
            # (n_data, n_dist_support, n_dist_support, n_interval_steps) = (data, from, to, interval)
            running_interval_class_distribution = torch.einsum("fti,di,df->dt",
                                                               self.transition_matrix,
                                                               interval_step_log_probs.exp(),
                                                               running_interval_class_distribution)
            # update cumulative
            cum_path_length_prob = cum_path_length_prob + step_prob
            if n >= self.effective_min_iterations and np.allclose(cum_path_length_prob.data.numpy(), 1):
                print(f"break after {n+1} iterations")
                break
        else:
            with np.printoptions(threshold=np.inf):
                print(f"cum_path_length_prob: {cum_path_length_prob.data.numpy()}")
                print(f"params: {self.get_params()}")
            raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")
        # add epsilon
        self.interval_class_distribution = self.interval_class_distribution + epsilon
        # normalise to account for border effects and path length cut-off
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)
        if np.any(np.isnan(self.interval_class_distribution.data.numpy())):
            print(self.interval_class_distribution)
            raise RuntimeWarning("got nan")


class FactorModel(DiffusionModel):

    def __init__(self,
                 max_iterations=100,
                 path_dist=Poisson,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.path_dist = path_dist
        self.dist_log_params = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise parameters
        if self.path_dist == Poisson:
            self.dist_log_params.data = torch.zeros((self.n_data, self.n_interval_steps), dtype=torch.float64)
        else:
            self.dist_log_params.data = torch.ones((self.n_data,
                                                    self.n_interval_steps,
                                                    2), dtype=torch.float64)

    def get_interpretable_params(self):
        return self.dist_log_params.exp().data.numpy().reshape(self.dist_log_params.shape[0], -1).copy()

    def perform_diffusion(self):
        # initialise distribution
        new_interval_class_distribution = self.init_interval_class_distribution
        # ensure finite DKL
        epsilon = 1e-50
        # apply different interval steps successively
        for interval_idx, step_length in enumerate(self.interval_steps):
            # path length distribution
            if self.path_dist == Poisson:
                path_length_dist = Poisson(rate=self.dist_log_params[:, interval_idx].exp())
                def path_length_dist_func(n):
                    return path_length_dist.log_prob(n).exp()
            elif self.path_dist == Binomial:
                total_count = self.dist_log_params[:, interval_idx, 0].exp()
                total_count_floor = total_count.floor()
                total_count_ceil = total_count.ceil()
                alpha = total_count - total_count_floor
                probs = self.dist_log_params[:, interval_idx, 1].sigmoid()
                floor_bin = Binomial(total_count=total_count_floor, probs=probs)
                ceil_bin = Binomial(total_count=total_count_ceil, probs=probs)
                def path_length_dist_func(n):
                    return alpha * ceil_bin.log_prob(n).exp() + (1 - alpha) * floor_bin.log_prob(n).exp()
            else:
                raise RuntimeWarning("Failed Case")
            # cumulative probability weight for termination condition
            cum_path_length_prob = torch.zeros(self.n_data, dtype=torch.float64)
            # marginalise latent variable
            for n in range(self.max_iterations):
                # probability to reach this step
                step_prob = path_length_dist_func(torch.tensor(n, dtype=torch.float64))
                # which values are within bounds for original and shifted distribution
                if n == 0:
                    # new dist for accumulation
                    old_interval_class_distribution = new_interval_class_distribution
                    new_interval_class_distribution = step_prob[:, None] * old_interval_class_distribution
                else:
                    if step_length > 0:
                        orig_slice = slice(n * step_length, None)
                        shifted_slice = slice(None, -n * step_length)
                    else:
                        orig_slice = slice(None, n * step_length)
                        shifted_slice = slice(-n * step_length, None)
                    # update output distributions (marginalise path length)
                    new_interval_class_distribution[:, orig_slice] = \
                        new_interval_class_distribution[:, orig_slice] + \
                        step_prob[:, None] * old_interval_class_distribution[:, shifted_slice]
                # update cumulative
                cum_path_length_prob = cum_path_length_prob + step_prob
                if np.allclose(cum_path_length_prob.data.numpy(), 1):
                    print(f"break after {n+1} iterations")
                    break
            else:
                with np.printoptions(threshold=np.inf):
                    print(f"cum_path_length_prob: {cum_path_length_prob.data.numpy()}")
                    print(f"params: {self.get_params()}")
                raise RuntimeWarning(f"maximum number of iterations ({self.max_iterations}) reached")
        self.interval_class_distribution = new_interval_class_distribution + epsilon
        # normalise to account for border effects and path length cut-off
        self.interval_class_distribution = self.interval_class_distribution / \
                                           self.interval_class_distribution.sum(dim=1, keepdim=True)
        if np.any(np.isnan(self.interval_class_distribution.data.numpy())):
            print(self.interval_class_distribution)
            raise RuntimeWarning("got nan")


class StaticDistributionModel(TonnetzModel):

    def __init__(self,
                 max_iterations=1000):
        super().__init__()
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise distribution with mode at reference center
        self.interval_class_log_distribution.data = torch.from_numpy(
            -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 5) ** 2
        )[None, :]

    def get_interpretable_params(self):
        return [[] for _ in range(self.n_data)]

    def _loss(self):
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        return super()._loss()


class GaussianModel:

    def __init__(self):
        self.data = None
        self.mean = None
        self.var = None
        self.distributions = None
        self.loss = None

    def set_data(self, data, weights=None):
        self.data = data
        pos = np.arange(data.shape[1])
        self.mean = (data * pos).sum(axis=1)
        self.var = (data * (pos[None, :] - self.mean[:, None]) ** 2).sum(axis=1)
        self.distributions = np.exp(-(pos[None, :] - self.mean[:, None]) ** 2 / self.var[:, None] / 2)
        self.distributions /= self.distributions.sum(axis=1, keepdims=True)
        self.loss = np.where(self.data == 0,
                             np.zeros_like(self.data),
                             self.data * (np.log(self.data) - np.log(self.distributions))).sum(axis=1)

    def get_results(self):
        """
        Compute model loss and return internal results
        :return: tuple of arrays (matched distribution, loss, center) containing results per data point
        """
        # return internal results
        return (self.distributions.copy(),
                self.loss.copy(),
                self.mean.copy())

    def get_interpretable_params(self):
        return np.sqrt(self.var)[:, None]
