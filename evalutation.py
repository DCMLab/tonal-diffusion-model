import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from tonal_diffusion_model import TonalDiffusionModel, GaussianModel, StaticDistributionModel

if __name__ == "__main__":
    # fix random seed
    np.random.seed(0)
    # generate data
    n_data = 10
    n_pitch_classes = 40
    mean = np.random.randint(10, 31, n_data)
    std = np.random.uniform(1, 5, n_data)
    data = np.exp(-((np.arange(n_pitch_classes)[None, :] - mean[:, None]) / std[:, None]) ** 2 / 2)
    data /= data.sum(axis=1, keepdims=True)

    # set up models
    diffusion_model = TonalDiffusionModel()
    static_model = StaticDistributionModel()
    gaussian_model = GaussianModel()
    # set data
    diffusion_model.set_data(data)
    static_model.set_data(data)
    gaussian_model.set_data(data)

    # optimise parameters
    # for model, n_params in [(diffusion_model, diffusion_model.n_data * diffusion_model.n_interval_steps),
    #                         (static_model, static_model.n_dist_support)]:
    for model, params in [(diffusion_model, diffusion_model.log_interval_step_weights.data.numpy()),
                          (static_model, static_model.static_interval_log_distribution.data.numpy())]:
        ret = minimize(fun=model.loss,
                       jac=model.grad,
                       # x0=np.zeros(n_params),
                       x0=params,
                       method='L-BFGS-B',
                       tol=1e-5,
                       # tol=1e-2,
                       # tol=1e1,
                       callback=model.callback)
        print(ret)

    # set up plot
    n_plots = 4
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))

    # plot data
    ax = axes[0]
    for d, c in zip(data, mean):
        ax.plot(d, '-o', label=f"{c}")
    ax.legend()

    # plot model
    for ax, mod, name in [(axes[1], diffusion_model, "Diffusion Model"),
                          (axes[2], static_model, "Static Model"),
                          (axes[3], gaussian_model, "Gaussian Model")]:
        dist = mod.get_distributions()
        loss = mod.get_loss()
        centers = mod.get_centers()
        for d, l, c in zip(dist, loss, centers):
            ax.plot(d, '-o', label=f"{round(c, 1)} | {np.format_float_scientific(l, 2)}")
        ax.legend()
        ax.set_title(name)

    plt.show()
