import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial
from torch.distributions.negative_binomial import NegativeBinomial
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
                 latent_shape=(),
                 soft_max_posterior=False,
                 separate_parameters=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.interval_steps = np.array(interval_steps)
        self.n_interval_steps = len(interval_steps)
        self.margin = margin
        self.n_data = None
        self.n_interval_classes = None
        self.n_dist_support = None
        self.latent_shape = latent_shape
        self.soft_max_posterior = soft_max_posterior
        self.separate_parameters = separate_parameters
        self.reference_center = None
        self.n_shifts = None
        self.interval_class_distribution = None
        self.data = None
        self.matched_dist = None
        self.matched_loss = None
        self.matched_shift = None
        self.matched_latent = None
        self.data_weights = None
        self.neg_log_likes = None
        if self.soft_max_posterior:
            self.beta = Parameter(torch.tensor([]))

    def _loss(self):
        self.match_distributions()
        if self.soft_max_posterior:
            # compute posterior of latent variables
            latent_dims = tuple(range(1, len(self.neg_log_likes.shape)))  # including transposition/shift!
            latent_log_posterior = -self.neg_log_likes * self.beta.exp()
            latent_log_posterior = latent_log_posterior - latent_log_posterior.logsumexp(dim=latent_dims, keepdim=True)
            # compute marginal neg-log-likelihood (per piece)
            neg_log_like = -(-self.neg_log_likes + latent_log_posterior).logsumexp(dim=latent_dims)
        else:
            data_slice = np.array(list(range(self.n_data)))
            if self.matched_latent is None:
                neg_log_like = self.neg_log_likes[data_slice, self.matched_shift]
            else:
                neg_log_like = self.neg_log_likes[(data_slice, self.matched_shift) + tuple(self.matched_latent)]
        if self.data_weights is None:
            return neg_log_like.mean()
        else:
            return (neg_log_like * self.data_weights).sum()

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # get necessary support of distribution and reference center
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.reference_center = None
            self.n_dist_support = self.n_interval_classes
            self.n_shifts = self.n_interval_classes
        else:
            self.n_dist_support = int(np.ceil((2 * self.margin + 1) * self.n_interval_classes))
            self.reference_center = int(np.round((self.margin + 0.5) * self.n_interval_classes))
            self.n_shifts = self.n_dist_support - self.n_interval_classes + 1
        if self.soft_max_posterior:
            self.beta.data = torch.tensor([0.])

    def get_interpretable_params(self, *args, **kwargs):
        d = dict(
            # loss=self.matched_loss.data.numpy().copy(),
            shift=self.matched_shift
        )
        if self.soft_max_posterior:
            d = dict(**d, beta=[self.beta.exp().data.numpy()[0] for _ in range(self.n_data)])
        if self.matched_latent is not None:
            d = dict(**d, **{f"latent_{idx+1}": latent for idx, latent in enumerate(self.matched_latent)})
        return d


    def match_distributions(self):
        if hasattr(self, "n_shifts"):
            n_shifts = self.n_shifts
        else:
            n_shifts = self.n_dist_support - self.n_interval_classes + 1
        all_data_indices = np.arange(self.n_data)
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.neg_log_likes = self.dkl(self.data[:, :, None], self.interval_class_distribution[:, :, :], dim=1)
            self.matched_shift = np.argmin(self.neg_log_likes.data.numpy(), axis=1)
            self.matched_dist = self.interval_class_distribution[all_data_indices, :, self.matched_shift]
            self.matched_loss = self.neg_log_likes[all_data_indices, self.matched_shift]
        else:
            self.neg_log_likes = torch.zeros((self.n_data, n_shifts) + self.latent_shape, dtype=torch.float64)
            latent_none = tuple(None for _ in self.latent_shape)
            latent_slice = tuple(slice(None) for _ in self.latent_shape)
            for shift in range(n_shifts):
                # keep profile dimension
                dist = self.interval_class_distribution[(slice(None),
                                                         slice(shift, shift + self.n_interval_classes)) +
                                                        latent_slice]
                dist = dist / dist.sum(dim=1, keepdim=True)
                # neg-log-likelihoods for all profiles (and all data as in parent function)
                nll = self.dkl(self.data[(slice(None), slice(None)) + latent_none], dist, dim=1)
                self.neg_log_likes[(slice(None), shift) + latent_slice] = nll
                if self.latent_shape:
                    matched_latent = np.unravel_index(
                        np.argmin(nll.data.numpy().reshape(self.n_data, -1), axis=1),
                        self.latent_shape
                    )
                    matched_dist = dist[(0, slice(None)) + matched_latent].transpose(0, 1)
                else:
                    matched_latent = ()
                    matched_dist = dist
                matched_data_indices = (all_data_indices,) + matched_latent
                if shift == 0:
                    self.matched_dist = matched_dist
                    self.matched_loss = nll[matched_data_indices]
                    self.matched_shift = np.zeros(self.n_data, dtype=int)
                    if self.latent_shape:
                        self.matched_latent = matched_latent
                else:
                    cond = nll[matched_data_indices] < self.matched_loss
                    self.matched_dist = torch.where(cond[:, None], matched_dist, self.matched_dist)
                    self.matched_loss = torch.where(cond, nll[matched_data_indices], self.matched_loss)
                    self.matched_shift = np.where(cond, shift, self.matched_shift)
                    if self.latent_shape:
                        self.matched_latent = np.where(cond, matched_latent, self.matched_latent)

    def get_results(self, *args, **kwargs):
        """
        Compute model loss and return internal results
        :return: tuple of arrays (matched distribution, loss, center) containing results per data point
        """
        # compute everything
        self._loss()
        # return internal results
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            return (self.matched_dist.data.numpy().copy(),
                    self.matched_loss.data.numpy().copy(),
                    self.matched_shift)
        else:
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
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.init_interval_class_distribution = np.zeros((self.n_data,
                                                              self.n_interval_classes,
                                                              self.n_interval_classes))
            self.init_interval_class_distribution[:,
            np.arange(self.n_interval_classes),
            np.arange(self.n_interval_classes)] = 1
        else:
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

    def perform_diffusion(self):
        raise NotImplementedError

    def _loss(self):
        self.perform_diffusion()
        return super()._loss()


class TonalDiffusionModel(DiffusionModel):

    def __init__(self,
                 min_iterations=None,
                 max_iterations=1000,
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
            if hasattr(self, "separate_parameters") and self.separate_parameters:
                self.effective_min_iterations = int(np.ceil(
                    self.n_interval_classes / min(largest_step_up, largest_step_down)
                )) + 1
            else:
                self.effective_min_iterations = int(np.ceil(max(
                    self.reference_center / largest_step_down,
                    (self.n_dist_support - self.reference_center) / largest_step_up
                ))) + 1
        else:
            self.effective_min_iterations = self.min_iterations
        # initialise weights
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            self.log_interval_step_weights.data = torch.zeros((self.n_data, self.n_interval_steps, self.n_shifts),
                                                              dtype=torch.float64)
        else:
            self.log_interval_step_weights.data = torch.zeros((self.n_data, self.n_interval_steps),
                                                              dtype=torch.float64)
        # initialise distribution parameters
        # default values
        if self.path_dist in [Poisson, Geometric]:
            default_params = [0]
            # np.torch.zeros(self.n_data, dtype=torch.float64)
        elif self.path_dist in [Gamma, Binomial, NegativeBinomial]:
             # default_params = np.ones(2, dtype=np.float64)
             if self.path_dist == Gamma:
                 default_params = [2, -2]
                 # default_params[:, 0] *= 2
                 # default_params[:, 1] *= -2
             elif self.path_dist == Binomial:
                 default_params = [2, 0]
                 # default_params[:, 0] *= 2
                 # default_params[:, 1] *= 0
             else:
                 default_params = [0, 0]
                 # default_params *= 0
             # self.path_log_params.data = torch.from_numpy(default_params)
        else:
            raise RuntimeWarning("Failed Case")
        default_params = np.array(default_params, dtype=np.float64)
        # fill
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            full_params = np.zeros((self.n_data, len(default_params), self.n_shifts), dtype=np.float64)
            full_params[...] = default_params[None, :, None]
        else:
            full_params = np.zeros((self.n_data, len(default_params)), dtype=np.float64)
            full_params[...] = default_params[None, :]
        self.path_log_params.data = torch.from_numpy(full_params)

    def get_results(self, shifts=None, *args, **kwargs):
        ret = super().get_results(*args, **kwargs)
        if shifts is None:
            return ret
        else:
            return (self.interval_class_distribution[np.arange(self.n_data), :, shifts].data.numpy().copy(),
                    self.neg_log_likes[np.arange(self.n_data), shifts].data.numpy().copy(),
                    shifts)

    def get_interpretable_params(self, shifts=None, *args, **kwargs):
        if shifts is None:
            shifts = self.matched_shift
        # init from super
        d = super().get_interpretable_params()
        # select correct parameters in case of separate parameters
        if hasattr(self, "separate_parameters") and self.separate_parameters:
            path_log_params = self.path_log_params[np.arange(self.n_data), :, shifts]
            log_interval_step_weights = self.log_interval_step_weights[np.arange(self.n_data), :, shifts]
        else:
            log_interval_step_weights = self.log_interval_step_weights
            path_log_params = self.path_log_params
        # compute and add normalised weights
        weight_sum = log_interval_step_weights.exp().sum(dim=1).data.numpy()
        weights = log_interval_step_weights.exp().data.numpy() / weight_sum[:, None]
        d = dict(**d, weights=weights)
        # format path parameters
        if self.path_dist == Poisson:
            d = dict(**d, rate=path_log_params.exp().data.numpy().copy())
        elif self.path_dist == Geometric:
            d = dict(**d, probs=path_log_params.sigmoid().data.numpy().copy())
        elif self.path_dist == Gamma:
            d = dict(**d, concentration=path_log_params[:, 0].exp().data.numpy().copy(),
                     rate=path_log_params[:, 1].exp().data.numpy().copy())
        elif self.path_dist in [Binomial, NegativeBinomial]:
            d = dict(**d,
                     total_count = path_log_params[:, 0].exp().data.numpy().copy(),
                     probs=path_log_params[:, 1].sigmoid().data.numpy().copy())
        else:
            raise RuntimeWarning("Failed Case")
        return d

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
        elif self.path_dist == NegativeBinomial:
            total_count = self.path_log_params[:, 0].exp()
            probs = self.path_log_params[:, 1].sigmoid()
            path_length_dist = NegativeBinomial(total_count=total_count, probs=probs)
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
        cum_path_length_prob = None
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
            if hasattr(self, "separate_parameters") and self.separate_parameters:
                # update output distributions (marginalise path length)
                self.interval_class_distribution = self.interval_class_distribution + \
                                                   step_prob[:, None, :] * running_interval_class_distribution
                # perform transition (marginalise interval classes)
                # intermediate tensor has dimensions:
                # (n_data, n_dist_support, n_dist_support, n_interval_steps, n_shifts) = (data, from, to, interval, shift)
                running_interval_class_distribution = torch.einsum("fti,dis,dfs->dts",
                                                                   self.transition_matrix,
                                                                   interval_step_log_probs.exp(),
                                                                   running_interval_class_distribution)
            else:
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
            if cum_path_length_prob is None:
                cum_path_length_prob = torch.zeros_like(step_prob)
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

    def get_interpretable_params(self, *args, **kwargs):
        d = super().get_interpretable_params()
        return dict(**d,
                    dist_params=self.dist_log_params.data.numpy().reshape(self.dist_log_params.shape[0], -1).copy())

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


class SimpleStaticDistributionModel(TonnetzModel):

    def __init__(self,
                 max_iterations=1000):
        super().__init__()
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise distribution with mode at reference center
        self.interval_class_log_distribution.data = torch.from_numpy(
            -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 10) ** 2
        )[None, :]

    def get_interpretable_params(self, *args, **kwargs):
        return dict()

    def match_distributions(self):
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        super().match_distributions()


class StaticDistributionModel(TonnetzModel):

    def __init__(self,
                 n_profiles=1,
                 max_iterations=1000,
                 *args, **kwargs):
        super().__init__(latent_shape=(n_profiles,), *args, **kwargs)
        self.max_iterations = max_iterations
        self.interval_class_log_distribution = Parameter(torch.tensor([]))

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        # initialise distribution with mode at reference center
        log_dist = -((np.arange(self.n_dist_support) - self.reference_center) / self.n_dist_support * 10) ** 2
        # add some noise for the different profiles
        all_log_dists = np.random.uniform(-1e-3, 1e-3, (self.n_dist_support, self.latent_shape[0]))
        all_log_dists += log_dist[:, None]
        self.interval_class_log_distribution.data = torch.from_numpy(all_log_dists[None, :])

    def match_distributions(self):
        self.interval_class_distribution = self.interval_class_log_distribution.exp()
        super().match_distributions()


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
        return dict(mean=self.mean.copy(), std=np.sqrt(self.var))
