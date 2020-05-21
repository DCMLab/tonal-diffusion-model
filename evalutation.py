import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from tonal_diffusion_model import TonalDiffusionModel, GaussianModel

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
    model = TonalDiffusionModel()
    baseline = GaussianModel()
    model.set_data(data)
    baseline.set_data(data)

    # optimise parameters
    ret = minimize(fun=model.loss,
                   jac=model.grad,
                   x0=np.zeros(model.n_data * model.n_interval_steps),
                   method='L-BFGS-B',
                   tol=1e-5,
                   # tol=1e1,
                   callback=model.callback)
    print(ret)

    # set up plot
    n_plots = 3
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 7, 5))

    # plot data
    ax = axes[0]
    for d, c in zip(data, mean):
        ax.plot(d, '-o', label=f"{c}")
    ax.legend()

    # plot model
    for ax, mod in [(axes[1], model), (axes[2], baseline)]:
        dist = mod.get_distributions()
        loss = mod.get_loss()
        centers = mod.get_centers()
        for d, l, c in zip(dist, loss, centers):
            ax.plot(d, '-o', label=f"{round(c, 1)} | {np.format_float_scientific(l, 2)}")
        ax.legend()

    plt.show()
