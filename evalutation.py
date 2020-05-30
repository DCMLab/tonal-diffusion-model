import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from torch.optim import LBFGS, Adam

from tonal_diffusion_model import TonalDiffusionModel, GaussianModel, StaticDistributionModel

if __name__ == "__main__":

    # url = "https://raw.githubusercontent.com/DCMLab/TP3C/master/tpc_corpus.tsv"
    # df = pd.read_csv(url, sep="\t")
    # df.to_csv("data.tsv", index=False)
    df = pd.read_csv("data.tsv")
    # print(df.columns)

    # select pieces (tpc counts + metadata)
    bach = df[(df.composer == "Bach") & (df.display_year == 1722)]
    beethoven = df[(df.composer == "Beethoven") & (df.work_group == "Piano Sonatas") & (df.mov == "1.0")]
    liszt = df[(df.composer == "Liszt") & (df.display_year >= 1880)]

    # data to use
    raw_data = pd.concat([
        bach,
        beethoven,
        liszt,
    ])
    # print(raw_data)

    # get counts and normalise
    counts = raw_data.iloc[:, 12:].values
    data = counts / counts.sum(axis=1, keepdims=True)

    # get pitch class labels
    labels = raw_data.columns[12:].values
    # print(labels)

    # get piece title
    titles = [f"{cf} {cl} {wg} {wc} {op} {no} {mo} {ti}" for
              (cf,
               cl,
               wg,
               wc,
               op,
               no,
               mo,
               ti) in zip(raw_data['composer_first'],
                          raw_data['composer'],
                          raw_data['work_group'],
                          raw_data['work catalogue'],
                          raw_data['opus'],
                          raw_data['no'],
                          raw_data['mov'],
                          raw_data['title'])]
    # print(titles)

    # set up models
    diffusion_model = TonalDiffusionModel()
    static_model = StaticDistributionModel()
    gaussian_model = GaussianModel()

    # set data
    diffusion_model.set_data(data)
    static_model.set_data(data)
    gaussian_model.set_data(data)

    # optimise parameters
    scipy_optimize = True
    if scipy_optimize:
        for model, params in [(diffusion_model, diffusion_model.log_interval_step_weights.data.numpy()),
                              (static_model, static_model.static_interval_log_distribution.data.numpy())]:
            ret = minimize(fun=model.loss,
                           jac=model.grad,
                           x0=params,
                           method='L-BFGS-B',
                           tol=1e-5,
                           # tol=1e-2,
                           # tol=1e1,
                           callback=model.callback)
            print(ret)
    else:
        for model in [diffusion_model, static_model]:
            optimizer = Adam(params=model.parameters(), lr=1e-1)
            loss = []
            delta_it = 50
            delta_loss = 1e-3
            for it in range(1000):
                loss.append(optimizer.step(closure=lambda: model.closure()))
                print(f"iteration {it}")
                print(f"    loss: {loss[-1]}")
                if it > delta_it and loss[-delta_it] - loss[-1] < delta_loss:
                    break
            plt.plot(loss, '-o')
            plt.show()

    # set up plots
    n_plots = data.shape[0]
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), gridspec_kw=dict(left=0.05,
                                                                                     right=0.95,
                                                                                     bottom=0.001,
                                                                                     top=0.999))

    # plot data
    for idx in range(n_plots):
        ax = axes[idx]
        ax.plot(data[idx], '-o', label=f"empirical")
        ax.set_title(titles[idx])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

    # plot models
    loss_df = None
    for mod, name in [(diffusion_model, "Diffusion Model"),
                      (static_model, "Static Model"),
                      (gaussian_model, "Gaussian Model")]:
        dist = mod.get_distributions()
        loss = mod.get_loss()
        centers = mod.get_centers()
        params = mod.get_interpretable_params()
        df = pd.DataFrame(data=loss, columns=['loss'])
        df["model"] = name
        print(name)
        if loss_df is None:
            loss_df = df
        else:
            loss_df = pd.concat([loss_df, df])
        for idx in range(n_plots):
            ax = axes[idx]
            p = ", ".join([np.format_float_scientific(x, 2) for x in params[idx]])
            ax.plot(dist[idx], '-o', label=f"{name} [{p}] ({np.format_float_scientific(loss[idx], 2)})")
            ax.legend()

    fig.savefig("pieces.pdf")
    # plt.show()

    # box plots
    fig_box, ax_box = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(x="model", y="loss", data=loss_df, ax=ax_box)
    sns.swarmplot(x="model", y="loss", data=loss_df, ax=ax_box, color=".25")
    fig_box.savefig("box_plots.pdf")
