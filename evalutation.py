import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
from torch.distributions.poisson import Poisson
from torch.distributions.binomial import Binomial
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.geometric import Geometric
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from torch.optim import Adam
from optimization import WarmAdam

from tonal_diffusion_model import TonalDiffusionModel, GaussianModel, StaticDistributionModel, FactorModel, SimpleStaticDistributionModel

if __name__ == "__main__":

    compact_data = True
    # do_plot_all_pieces = True
    do_plot_all_pieces = False
    do_plot_single_pieces = True
    # do_plot_single_pieces = False
    # do_plot_draft = True
    do_plot_draft = False
    # do_train_models = True
    do_train_models = False
    scipy_optimize = False
    do_separate_composers = True
    # soft_max = None
    soft_max = True
    # soft_max = False
    suffix = ""
    # do_show_learning_curves = True
    do_show_learning_curves = False
    do_save_learning_curves = True
    # do_save_learning_curves = False
    allow_model_rename = True
    # allow_model_rename = False
    # plot_empirical_as_background = True
    plot_empirical_as_background = False
    # plot_params = True
    plot_params = False

    def set_plot_style():
        # plt.style.use("ggplot")
        sns.set_style("whitegrid")
        # sns.set_palette(sns.hls_palette(5, l=.3, s=.8))
        # sns.set_palette("deep")
        # sns.set_palette(sns.color_palette(["#4C07DE",
        #                                    "#12C4B5",
        #                                    "#A6DE1B",
        #                                    "#E67C02",
        #                                    "#CC1A02"]))
        sns.set_palette(sns.color_palette(["#8210E0",
                                           "#0091EB",
                                           "#78CF08",
                                           "#E3A00B",
                                           "#C71B08"]))

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

    # select single piece
    # raw_data = raw_data[raw_data['filename'] == "210606-Prelude_No._1_BWV_846_in_C_Major.mxl.csv"]

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

    # set up models
    all_models = [
        # (FactorModel(path_dist=Poisson), "Factor Model (Poisson)"),
        # (FactorModel(path_dist=Binomial), "Factor Model (Binomial)"),
        #
        # (TonalDiffusionModel(path_dist=Geometric, soft_max_posterior=False if soft_max is None else soft_max),
        #  "Diffusion Model (Geometic)"),
        # (TonalDiffusionModel(path_dist=NegativeBinomial, soft_max_posterior=False if soft_max is None else soft_max),
        #  "Diffusion Model (Neg. Binomial)",
        #  dict(lr=4e-1, init_lr=4e-1, lr_beta=0.95)),
        #
        (StaticDistributionModel(n_profiles=1, soft_max_posterior=True if soft_max is None else soft_max),
         "Static (1 profile)", "Static\n(1 profile)",
         dict(lr=2e-1)),
        (StaticDistributionModel(n_profiles=2, soft_max_posterior=True if soft_max is None else soft_max),
         "Static (2 profiles)", "Static\n(2 profiles)",
         dict(lr=2e-1)),
        (TonalDiffusionModel(path_dist=Binomial, soft_max_posterior=False if soft_max is None else soft_max,
                             interval_steps=(-1, 1)),
         "TDM (Binomial, 1D)", "TDM\n(Binomial, 1D)",
         dict(lr=4e-2, init_lr=1e-2, lr_beta=0.95)),
        (TonalDiffusionModel(path_dist=Poisson, soft_max_posterior=False if soft_max is None else soft_max),
         "TDM (Poisson)", "TDM\n(Poisson)",
         dict(lr=2e-1)),
        (TonalDiffusionModel(path_dist=Binomial, soft_max_posterior=False if soft_max is None else soft_max),
         "TDM (Binomial)", "TDM\n(Binomial)",
         dict(defaults=dict(lr=5e-2, init_lr=1e-2, lr_beta=0.95), beta=dict(lr=1e-2, init_lr=1e-2, lr_beta=0.99))),
        #
        # (StaticDistributionModel(n_profiles=3), "Static Model (3 profiles)"),
        # (SimpleStaticDistributionModel(), "Simple Static Model"),
    ]

    # set data
    for model, _, _, params in all_models:
        model.set_data(data=data, weights=raw_data['weights'].values)

    # optimise parameters or load models
    for model_idx, (model, name, label, params) in enumerate(all_models):
        model_file = "model " + name + suffix + ".tar"
        learning_curve_plot_file = "model " + name + suffix + " learning.pdf"
        if do_train_models:
            if scipy_optimize:
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
                print(name)
                best_model_params = None
                best_loss = np.inf
                loss = []
                # collect parameters to optimise and potential custom values for learning rate etc
                opt_params = []
                for x in model.parameters():
                    d = {'params': x}
                    # check is custom value was provided for this parameter
                    for key, val in params.items():
                        if hasattr(model, key) and getattr(model, key) is x:
                            d = {**d, **val}
                    opt_params.append(d)
                # init optimiser with default parameters (which are overwritten by any parameter-specific values)
                optimizer = WarmAdam(
                    params=opt_params,
                    **params['defaults']
                )
                delta_it = 100
                delta_loss = 1e-4
                # delta_it = 50
                # delta_loss = 1e-3
                # delta_it = 10
                # delta_loss = 1e-1
                do_break = False
                # do_break = True
                for it in range(10000):
                    current_loss = optimizer.step(closure=lambda: model.closure())
                    loss.append(current_loss)
                    print(f"iteration {it}")
                    print(f"    loss: {current_loss}")
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_params = model.get_params()
                    if do_break or (it > delta_it and loss[-delta_it] - loss[-1] < delta_loss):
                        break
                print(f"best model is final model? "
                      f"{np.array_equal(all_models[model_idx][0].get_params(), best_model_params)}")
                model.set_params(best_model_params)
                print(f"best model saved? "
                      f"{np.array_equal(all_models[model_idx][0].get_params(), best_model_params)}")
                print(name)
                if do_show_learning_curves or do_save_learning_curves:
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(loss, '-o')
                    if do_save_learning_curves:
                        fig.savefig(learning_curve_plot_file)
                    if do_show_learning_curves:
                        plt.show()
            # save model
            print(f"saving model '{model_file}'...")
            torch.save(all_models[model_idx], model_file)
            print("DONE")
        else:
            print(f"loading model '{model_file}'...")
            try:
                (old_model, old_name, old_label, old_params) = torch.load(model_file)
            except ValueError:
                # old model files
                (old_model, old_name, old_params) = torch.load(model_file)
            if allow_model_rename:
                all_models[model_idx] = (old_model, name, label, old_params)
            else:
                all_models[model_idx] = (old_model, old_name, old_params)
            print("DONE")

    # set up plots and plot data
    set_plot_style()
    x = np.arange(data.shape[1])
    if do_plot_all_pieces or do_plot_single_pieces:
        print("set up plots and plot data...")
        n_plots = data.shape[0]
        if plot_empirical_as_background:
            width = 15
        else:
            width = 10
        height = 5
        if do_plot_all_pieces:
            fig_all_pieces, axes_all_pieces = plt.subplots(n_plots, 1, figsize=(width, height * n_plots),
                                                           gridspec_kw=dict(left=0.05,
                                                                            right=0.95,
                                                                            bottom=0.001,
                                                                            top=0.999))
        figs_single_pieces = []
        axes_single_pieces = []
        for idx in range(n_plots):
            if do_plot_draft and idx > 0:
                break
            plot_axes = []
            if do_plot_all_pieces:
                plot_axes.append(axes_all_pieces[idx])
            if do_plot_single_pieces:
                f, a = plt.subplots(1, 1, figsize=(width, height))
                figs_single_pieces.append(f)
                axes_single_pieces.append(a)
                plot_axes.append(a)
            for ax in plot_axes:
                if plot_empirical_as_background:
                    # color = next(ax._get_lines.prop_cycler)['color']
                    color = (0, 0, 0)
                    facecolor = (1, 1, 1, 0.5)
                    edgecolor = (0, 0, 0)
                    # ax.plot(data[idx], '-o', label=f"empirical")
                    ax.fill_between(x, 0, data[idx], alpha=0.2, color=color)
                    ax.scatter(x, data[idx], s=15, label=f"empirical", facecolor=facecolor, edgecolor=edgecolor)
                else:
                    ax.fill_between(x, 0, data[idx], alpha=0.1, linewidth=0, color=(0, 0, 0))
                    ax.bar(x, data[idx], width=0.3, linewidth=0, color=(0, 0, 0, 0.6),
                           label="empirical")
                ax.tick_params(axis='x', rotation=90)
                ax.grid(False, axis='x')
                ax.set_title(titles[idx])
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlim(min(x), max(x))
                ax.legend()
        print("DONE")

    # plot models
    results_df = None
    print("plotting pieces and collecting statistics...")
    n_models = len(all_models)
    # width = 1 / (n_models + 1)
    width = 1 / n_models / 2
    for model_idx, (model, name, label, params) in enumerate(all_models):
        print(f"    {name}")
        print("    collect results...")
        (dist, loss, centers) = model.get_results()
        params = model.get_interpretable_params()
        # collect loss
        df = pd.DataFrame(data=loss, columns=['loss'])
        df["model"] = name
        df["model_label"] = label
        df["composer"] = raw_data['composer'].values
        df["piece"] = titles
        df["origin"] = centers
        print(centers)
        # df["origin TPC"] = labels[centers]
        for key, val in params.items():
            if isinstance(val, np.ndarray) and len(val.shape) == 2:
                for i in range(val.shape[1]):
                    df[f"{key}_{i}"] = val[:, i]
            else:
                df[key] = val
        if results_df is None:
            results_df = df
        else:
            results_df = pd.concat([results_df, df])
        print("    DONE")
        if do_plot_all_pieces or do_plot_single_pieces:
            print("    plotting...")
            for idx in range(n_plots):
                if do_plot_draft and idx > 0:
                    break
                if plot_params:
                    p = " [" + \
                        ", ".join([f"c: {centers[idx]}"] +
                                  [f"{key}: {np.round(val[idx], 2)}" for key, val in params.items()]) + \
                        "] "
                else:
                    p = " "
                new_x = x + (model_idx - n_models / 2) * width
                # ax.plot(dist[idx], '-o', label=f"{name} [{p}] ({np.format_float_scientific(loss[idx], 2)})")
                plot_axes = []
                if do_plot_all_pieces:
                    plot_axes.append(axes_all_pieces[idx])
                if do_plot_single_pieces:
                    plot_axes.append(axes_single_pieces[idx])
                for ax in plot_axes:
                    color = next(ax._get_lines.prop_cycler)['color']
                    if plot_empirical_as_background:
                        ax.bar(new_x, dist[idx], width=width, linewidth=0, color=color,
                               # label=f"{name} {p} ({np.format_float_scientific(loss[idx], 2)})",
                               label=f"{name}{p}[{np.round(loss[idx], 3)}]",
                               )
                        ax.plot(new_x, dist[idx], linewidth=0.3, color=color, solid_joinstyle='bevel')
                    else:
                        if name == "TDM (Binomial)":
                            linewidth = 2
                            # ax.fill_between(x, 0, dist[idx], color=color, alpha=0.1)
                        else:
                            linewidth = 1
                        ax.plot(x, dist[idx], '-o', linewidth=linewidth, markersize=4, color=color,
                                label=f"{name}{p}[{np.round(loss[idx], 3)}]")
                    ax.legend()
            print("    DONE")
    # calculate mean values and variances
    results_df['binomial_mean'] = results_df['total_count'].values * results_df['probs'].values
    results_df['binomial_std'] = np.sqrt(results_df['total_count'].values *
                                         results_df['probs'].values *
                                         (1 - results_df['probs'].values))
    results_df.to_csv("results.csv")
    if do_plot_all_pieces:
        print("    saving pieces in single plot...")
        fig_all_pieces.savefig("pieces" + suffix + ".pdf")
        print("    DONE")
    if do_plot_single_pieces:
        print("    saving pieces in separate plots...")
        os.makedirs("pieces" + suffix, exist_ok=True)
        for t, f in zip(raw_data['filename'], figs_single_pieces):
            f.tight_layout()
            f.savefig(f"pieces{suffix}/{t}.pdf")
        print("    DONE")
    print("DONE")


    # box plots
    print("plotting statistics...")
    set_plot_style()
    fig_box, ax_box = plt.subplots(1, 1, figsize=(1.1 * len(all_models), 3.5))
    box_markersize = 4
    swarm_markersize = 1.2
    box_kwargs = dict(linewidth=1,
                      showfliers=False,
                      showmeans=True,
                      meanline=True,
                      meanprops=dict(color=(0.8, 0, 0)))
    swarm_kwargs = dict(alpha=0.6)
    if do_separate_composers:
        swarm_kwargs = {**swarm_kwargs, **dict(hue="composer")}
        box_kwargs = {**box_kwargs, **dict(hue="composer")}
    sns.boxplot(x="model_label", y="loss", data=results_df, ax=ax_box,
                fliersize=box_markersize, **box_kwargs)
    sns.swarmplot(x="model_label", y="loss", data=results_df, ax=ax_box, dodge=True,
                  # color=".25",
                  color=(0, 0, 0, 0.1),
                  size=swarm_markersize, **swarm_kwargs, add_legend=False)
    ax_box.set_ylabel("cross-entropy")
    ax_box.set_xlabel(None)
    # ax_box.set_yscale('log')
    fig_box.tight_layout()
    fig_box.savefig("box_plots" + suffix + ".pdf")
    print("DONE")
