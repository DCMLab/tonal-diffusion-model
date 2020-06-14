import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions.poisson import Poisson
from torch.distributions.binomial import Binomial
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from torch.optim import Adam

from tonal_diffusion_model import TonalDiffusionModel, GaussianModel, StaticDistributionModel, FactorModel, SimpleStaticDistributionModel

if __name__ == "__main__":

    compact_data = True
    do_plot_pieces = True
    do_train_models = True
    scipy_optimize = False
    do_separate_composers = True

    if compact_data:
        df = pd.read_csv("tdm_data.tsv", sep='\t')

        # select pieces (tpc counts + metadata)
        bach = df[df.composer == "Bach"]
        beethoven = df[df.composer == "Beethoven"]
        liszt = df[df.composer == "Liszt"]
        scriabin = df[df.composer == "Scriabin"]
    else:
        # url = "https://raw.githubusercontent.com/DCMLab/TP3C/master/tpc_corpus.tsv"
        # df = pd.read_csv(url, sep="\t")
        # df.to_csv("data.tsv", index=False)
        df = pd.read_csv("data.tsv")

        # select pieces (tpc counts + metadata)
        bach = df[(df.composer == "Bach") & (df.display_year == 1722)]
        beethoven = df[(df.composer == "Beethoven") & (df.work_group == "Piano Sonatas") & (df.mov == "1.0")]
        liszt = df[(df.composer == "Liszt") & (df.display_year >= 1880)]
        scriabin = df[(df.composer == "Scriabin")]

    # data to use
    raw_data = pd.concat([
        bach,
        beethoven,
        liszt,
        # scriabin,
    ])

    # weight composers equally
    raw_data["weights"] = 0.
    for comp, w in raw_data['composer'].value_counts().iteritems():
        raw_data.loc[raw_data[raw_data['composer'] == comp].index, 'weights'] = 1 / w

    # get counts and normalise, get pitch class labels
    if compact_data:
        counts = raw_data.iloc[:, 2:-1].values
        labels = raw_data.columns[2:-1].values
    else:
        counts = raw_data.iloc[:, 12:-1].values
        labels = raw_data.columns[12:-1].values
    data = counts / counts.sum(axis=1, keepdims=True)

    # get piece title
    if compact_data:
        titles = [f"{cmp}: {fln}" for (cmp, fln) in zip(raw_data['composer'], raw_data['filename'])]
    else:
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

    if do_train_models:
        # set up models
        trainable_models = [
            # (FactorModel(path_dist=Poisson), "Factor Model (Poisson)"),
            # (FactorModel(path_dist=Binomial), "Factor Model (Binomial)"),
            (TonalDiffusionModel(path_dist=Poisson), "Diffusion Model (Poisson)"),
            (TonalDiffusionModel(path_dist=Binomial), "Diffusion Model (Binomial)"),
            (TonalDiffusionModel(path_dist=Binomial, interval_steps=(-1, 1)), "Diffusion Model (Binomial, 1D)"),
            (StaticDistributionModel(n_profiles=1), "Static Model (1 profile)"),
            (StaticDistributionModel(n_profiles=2), "Static Model (2 profiles)"),
            (StaticDistributionModel(n_profiles=3), "Static Model (3 profiles)"),
            # (SimpleStaticDistributionModel(), "Simple Static Model"),
        ]
        non_trainable_models = [
            # (GaussianModel(), "Gaussian Model"),
        ]
        all_models = trainable_models + non_trainable_models

        # set data
        for model, name in all_models:
            model.set_data(data=data, weights=raw_data['weights'].values)

        # optimise parameters
        if scipy_optimize:
            for model, name in trainable_models:
                ret = minimize(fun=model.loss,
                               jac=model.grad,
                               x0=model.get_params(),
                               method='L-BFGS-B',
                               # tol=1e-6,
                               tol=1e-5,
                               # tol=1e-2,
                               # tol=1e1,
                               callback=model.callback)
                print(ret)
        else:
            for model, name in trainable_models:
                print(name)
                loss = []
                # optimizer = Adam(params=model.parameters(), lr=1e-2)
                # delta_it = 100
                # delta_loss = 1e-5
                optimizer = Adam(params=model.parameters(), lr=1e-1)
                delta_it = 50
                delta_loss = 1e-3
                # optimizer = Adam(params=model.parameters(), lr=1e-1)
                # delta_it = 10
                # delta_loss = 5e-1
                do_break = False
                # do_break = True
                for it in range(10000):
                    loss.append(optimizer.step(closure=lambda: model.closure()))
                    print(f"iteration {it}")
                    print(f"    loss: {loss[-1]}")
                    if do_break or (it > delta_it and loss[-delta_it] - loss[-1] < delta_loss):
                        break
                print(name)
                # plt.plot(loss, '-o')
                # plt.show()

        # save models
        print("saving models...")
        torch.save({name: model for model, name in all_models},
                   "models.tar")
        print("DONE")
    else:
        print("loading models...")
        all_models = [(model, name) for name, model in torch.load("models.tar").items()]
        print("DONE")



    # set up plots and plot data
    sns.set_style("whitegrid")
    x = np.arange(data.shape[1])
    if do_plot_pieces:
        print("set up plots and plot data...")
        n_plots = data.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), gridspec_kw=dict(left=0.05,
                                                                                         right=0.95,
                                                                                         bottom=0.001,
                                                                                         top=0.999))
        for idx in range(n_plots):
            ax = axes[idx]
            # color = next(ax._get_lines.prop_cycler)['color']
            color = (0, 0, 0)
            facecolor = (1, 1, 1, 0.5)
            edgecolor = (0, 0, 0)
            ax.tick_params(axis='x', rotation=90)
            ax.grid(False, axis='x')
            # ax.plot(data[idx], '-o', label=f"empirical")
            ax.fill_between(x, 0, data[idx], alpha=0.2, color=color)
            ax.scatter(x, data[idx], s=15, label=f"empirical", facecolor=facecolor, edgecolor=edgecolor, zorder=10)
            ax.set_title(titles[idx])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlim(min(x), max(x))
        print("DONE")

    # plot models
    loss_df = None
    print("plotting pieces and collecting statistics...")
    n_models = len(all_models)
    # width = 1 / (n_models + 1)
    width = 1 / n_models / 2
    for model_idx, (model, name) in enumerate(all_models):
        print(name)
        (dist, loss, centers) = model.get_results()
        params = model.get_interpretable_params()
        df = pd.DataFrame(data=loss, columns=['loss'])
        df["model"] = name
        df["composer"] = raw_data['composer'].values
        df["piece"] = raw_data['filename'].values
        if loss_df is None:
            loss_df = df
        else:
            loss_df = pd.concat([loss_df, df])
        if do_plot_pieces:
            print("    plotting...")
            for idx in range(n_plots):
                ax = axes[idx]
                color = next(ax._get_lines.prop_cycler)['color']
                # p = ", ".join([np.format_float_scientific(x, 2) for x in params[idx]])
                p = " [" + ", ".join([str(np.round(x, 2)) for x in params[idx]]) + "] "
                p = ""
                new_x = x + (model_idx - n_models / 2) * width
                # ax.plot(dist[idx], '-o', label=f"{name} [{p}] ({np.format_float_scientific(loss[idx], 2)})")
                ax.bar(new_x, dist[idx], width=width, linewidth=0, color=color,
                       # label=f"{name} {p} ({np.format_float_scientific(loss[idx], 2)})",
                       label=f"{name}{p}({np.round(loss[idx], 2)})",
                       )
                ax.plot(new_x, dist[idx], linewidth=0.3, color=color, solid_joinstyle='bevel')
                ax.legend()
            print("    DONE")
    loss_df.to_csv("loss.csv")
    if do_plot_pieces:
        fig.savefig("pieces.pdf")
    print("DONE")


    # box plots
    print("plotting statistics...")
    sns.set_style("whitegrid")
    fig_box, ax_box = plt.subplots(1, 1, figsize=(3 * len(all_models), 6))
    swarm_markersize = 3
    box_markersize = 4
    swarm_mpl_kwargs = dict()
    if do_separate_composers:
        sns.boxplot(x="model", y="loss", hue="composer", data=loss_df, ax=ax_box, fliersize=box_markersize)
        sns.swarmplot(x="model", y="loss", hue="composer", data=loss_df, ax=ax_box, dodge=True, color=".25",
                      size=swarm_markersize, **swarm_mpl_kwargs, add_legend=False)
    else:
        sns.boxplot(x="model", y="loss", data=loss_df, ax=ax_box, fliersize=box_markersize)
        sns.swarmplot(x="model", y="loss", data=loss_df, ax=ax_box, dodge=True, color=".25",
                      size=swarm_markersize, **swarm_mpl_kwargs, add_legend=False)
    ax_box.set_ylabel("cross-entropy")
    ax_box.set_xlabel(None)
    fig_box.savefig("box_plots.pdf")
    print("DONE")
