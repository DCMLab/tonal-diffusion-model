import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pitchplots.static import tonnetz

import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import pandas as pd

import glob
from tqdm import tqdm

class Tone:

    ### Functions to generate arbitrary line-of-fifths (lof) segment
    steps = {s:idx for s, idx in zip(list('FCGDAEB'), range(7))}
    int_strings = ['+P5', '-P5', '+m3', '-m3', '+M3', '-M3']

    @staticmethod
    def get_lof_no(tpc):
        """Returns tpc from lof number"""
        step = Tone.steps[tpc[0]]
        accs = tpc[1:].count('#') - tpc[1:].count('b')

        return step + 7 * accs

    @staticmethod
    def get_tpc(lof_no):
        """Returns tpc for lof number"""
        a, b = divmod(lof_no, 7)
        d = {v:k for k, v in Tone.steps.items()}
        tpc = d[b]
        if a < 0:
            tpc += abs(a) * 'b'
        if a > 0:
            tpc += abs(a) * '#'
        return tpc

    @staticmethod
    def get_lof(min_tpc, max_tpc):
        """Returns number range on from tpcs"""
        min = Tone.get_lof_no(min_tpc)
        max = Tone.get_lof_no(max_tpc)

        if max < min:
            raise UserWarning(f"{min_tpc} is not lower than {max_tpc}.")
        return [ Tone.get_tpc(l) for l in np.arange(min, max + 1) ]

    ### Tonnetz plotting
    e_x = np.array([1, 0])
    e_y = np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)])
    eq = np.array([3, 1])

    @staticmethod
    def plot(tones, center, weights):
        weights = np.array(weights)
        weights /= np.max(weights)

        cmap = mpl.cm.get_cmap('Reds')

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        center_x, center_y = [t.loc for t in tones if t.name == center][0]
        x_min = center_x - 3
        x_max = center_x + 3
        y_min = center_y - 3
        y_max = center_y + 3
        for t, w in zip(tones, weights):
            for eq_idx in range(-3, 4):
                loc = t.get_loc(eq_idx)
                if not (x_min <= loc[0] <= x_max and y_min <= loc[1] <= y_max):
                    continue
                ax.text(*loc,
                        t.name,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=16)
                ax.add_patch(mpl.patches.CirclePolygon(
                    loc,
                    radius=np.tan(np.pi / 6),
                    resolution=6,
                    facecolor=cmap(w),
                    edgecolor=(0, 0, 0)))
        plt.axis('equal')
        plt.axis('off')
        return fig

    ### Diffusion

    intervals = [
        1,   # fifth up
        -1,  # fifth down
        -3,  # minor third up
        3,   # minor third down
        4,  # major third up
        -4    # major third down
    ]

    i = len(intervals)

    @staticmethod
    def diffuse(tones,
                center,
                action_probs,
                discount=None,
                init_dist=None,
                max_iter=10000,
                atol=1e-10,
                alpha=1,
                raise_on_max_iter=True,
                animate=False,
                normalize_action_probs=True,
                open_boundary=True):
        n = len(tones)
        # initialize init_dist if not provided or convert to numpy array
        if init_dist is None:
            init_dist = np.zeros(n)
            for idx, t in enumerate(tones):
                if t.name == center:
                    init_dist[idx] = 1.
        else:
            init_dist = np.array(init_dist)
        if not np.isclose(np.sum(init_dist), 1.):
            raise UserWarning(f"Init dist not normalized (sum={np.sum(init_dist)})")

        # convert action_probs to numpy array
        action_probs = np.array(action_probs)
        # normalize
        if normalize_action_probs:
            action_probs = action_probs / np.sum(action_probs)
        # construct transition matrix
        pi = np.zeros((n, n))
        for from_idx in range(n):
            for action_idx, step in enumerate(Tone.intervals):
                to_idx = from_idx + step
                if 0 <= to_idx < n:
                    # step is within bounds
                    pi[from_idx, to_idx] += action_probs[action_idx]
                elif not open_boundary:
                    # step would "leave" tonal range --> action has no effect
                    pi[from_idx, from_idx] += action_probs[action_idx]
        if not open_boundary:
            np.testing.assert_almost_equal(pi.sum(axis=1), 1)
        # diffuse
        current_dist = init_dist.copy()
        next_dist = np.zeros_like(init_dist)
        intermediate_dists = []
        for iteration in range(max_iter):
            next_dist = (1 - discount) * init_dist + discount * np.einsum('ij,i->j', pi, current_dist)
            if np.all(np.isclose(current_dist, next_dist, atol=atol)):
                break
            else:
                # use new distribution as current for next iteration
                current_dist = (1 - alpha) * current_dist + alpha * next_dist
                if animate:
                    intermediate_dists.append(current_dist)
        else:
            if raise_on_max_iter:
                raise UserWarning(f"Did not converge after {iteration} iterations")
        # check for approximate normalization
        if not open_boundary:
            norm = next_dist.sum()
            np.testing.assert_almost_equal(norm, 1)
        # normalize (for open boundary and to eliminate roundoff errors)
        next_dist /= next_dist.sum()
        if animate:
            return intermediate_dists
        return next_dist

    def __init__(self, loc, name, weight=0):
        self.loc = np.array(loc)
        self.name = name

    def get_loc(self, eq_idx=0):
        x, y = self.loc + Tone.eq * eq_idx
        return x * Tone.e_x + y * Tone.e_y

    @staticmethod
    def kl(p, q):
        return entropy(p,q, base=2)

    @staticmethod
    def jsd(p, q, base=2):
        ## convert to np.array
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        p, q = p/p.sum(), q/q.sum()
        m = 1./2*(p + q)
        return entropy(p,m, base=base)/2. + entropy(q, m, base=base)/2.

    @staticmethod
    def piece_freqs(csv_path, by_duration=True):
        # read piece and rename double sharps
        df = pd.read_csv(csv_path, index_col=0, engine='python')
        df['tpc'] = df['tpc'].str.replace('x', '##')

        ## normalize
        if by_duration:
            # by duration
            freqs = df.groupby('tpc')['duration'].sum()
            freqs /= freqs.sum()
        else:
            # by counts
            freqs = df.tpc.value_counts(normalize=True)

        # sort on line of fifths and determine most frequent tpc
        freqs = freqs.reindex(lof).fillna(0)
        center = freqs.idxmax()
        return freqs.values, center


if __name__ == "__main__":
    #
    # lof = Tone.get_lof('Fbb', 'B##')
    # tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]
    # weights = Tone.diffuse(tones=tones,
    #                        center="C",
    #                        action_probs=[0, 00, 0, 0, 0, 0],
    #                        discount=[0.1],
    #                        # atol=0.1,
    #                        max_iter=25,
    #                        raise_on_max_iter=False,
    #                        alpha=1,
    #                        animate=True)
    # for idx, w in enumerate(weights):
    #     fig = Tone.plot(tones, 'C', w)
    #     fig.tight_layout()
    #     file_name = f"animation_{str(idx).zfill(4)}.png"
    #     print(f"saving '{file_name}'")
    #     fig.savefig(file_name)
    #     plt.close(fig)
    # # create video by calling (adjust speed via framerate):
    # # ffmpeg -framerate 10 -pattern_type glob -i './animation_*.png' -c:v libx264 -r 30 -pix_fmt yuv420p animation.mp4
    # exit()

    lof = Tone.get_lof('Fbb', 'B##')
    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]
    dur = True

    ### Example pieces
    ex_pieces = glob.glob("data/*.csv")

    meta = pd.read_csv('../ExtendedTonality/metadata.csv', sep='\t', encoding='utf-8')
    meta = meta[meta.filename.notnull()]
    path = '..\\ExtendedTonality\\data\\DataFrames\\'
    csvs = [f for f in glob.glob(path+"*.csv")]
    pieces = []
    composers = []
    years = []
    for i, row in meta.iterrows():
        if  (path + row.filename + '.csv' in csvs):
            pieces.append(path + row.filename + '.csv')
            composers.append(row.composer)
            years.append(row.display_year)

    ### set fixed (initial) discount parameter for all intervals
    discount = .5
    ### INFERENCE
    # Constraint 1: weights and discounts must be between 0 and 1
    bnds = [(0, 1)] * Tone.i + [(0,1)] # 6 step directions plus discount
    # Constraint 2: sum of weights must be 1
    def con(x):
        return sum(x[:Tone.i]) - 1
    cons = {'type':'eq', 'fun': con}

    def cost_f(x, args):
        weights = Tone.diffuse(
                           tones=tones,
                           center=center,
                           action_probs=x[:6],
                           discount=x[6:],
                           open_boundary=False
                           )
        KL = Tone.kl(args, weights)
        return KL#Tone.jsd(weights, args)

    KLs = []
    best_ps = []
    for piece in tqdm(pieces):
    # for piece in [ex_pieces[i] for i in [0,2,11,19]]:
        freqs, center = Tone.piece_freqs(piece, by_duration=dur)

        mini = minimize(
            fun=cost_f,
            x0=[1/6] * Tone.i + [.9],
            args=(freqs),
            method="SLSQP", # Sequential Least SQares Programming
            bounds=bnds,
            constraints=cons
        )
        best_params = mini.get('x')

        best_weights = Tone.diffuse(tones=tones,
                                    center=center,
                                    action_probs=best_params[:Tone.i],
                                    discount=best_params[Tone.i:],
                                    animate=False)
        best_weights = np.array(best_weights)

        # for idx, w in enumerate(best_weights):
        #     fig = Tone.plot(tones, center, w)
        #     fig.tight_layout()
        #     file_name = f"animation_{str(idx).zfill(4)}.png"
        #     print(f"saving '{file_name}'")
        #     fig.savefig(file_name)
        #     plt.close(fig)

        # also plot actual distribution
        # fig = Tone.plot(tones, center, freqs)
        # fig.tight_layout()
        # fig.savefig('piece_dist.png')
        # plt.close()
        # create video by calling (adjust speed via framerate):
        # ffmpeg -framerate 10 -pattern_type glob -i './animation_*.png' -c:v libx264 -r 30 -pix_fmt yuv420p animation.mp4
        # exit()

        KLs.append(Tone.kl(freqs, best_weights))
        best_ps.append(best_params)

        ### PLOT
        # plot optimal parameters

        # fig, ax = plt.subplots(figsize=(6,6))
        # x = np.arange(best_params[:-6].shape[0])
        # ax.bar(x, best_params[:-6])
        # ds = [round(p,3) for p in best_params[-6:]]
        # plt.xticks(x, [f'{i}\n{ds[j]}'  for i, j in zip(Tone.int_strings, range(6))])
        # ax.tick_params(axis='both', which='both', labelsize=14)
        # # plt.title(piece)
        # plt.ylim(0,1)
        # plt.tight_layout()
        # plt.savefig(f'img/pieces/{piece[5:-4]}_best_params.png', dpi=300)
        # plt.show()

        # plot both distributions
        # pd.DataFrame(
        #     {'original':freqs, 'estimate':best_weights}
        #     ).plot(
        #         kind='bar',
        #         figsize=(12,6)
        #     )
        # plt.title(f"JSD: {round(Tone.jsd(freqs, best_weights), 3)}") # \n{piece}
        # plt.xticks(np.arange(len(lof)),lof)
        # plt.tight_layout()
        # # plt.savefig(f'img/pieces/{piece[5:-4]}_evaluation.png')
        # plt.show()
    #
    #
    #     # plot actual distribution (has to be adapted to include duration)
    #     df =pd.read_csv(piece)
    #     df['tpc'] = df['tpc'].str.replace('x', '##')
    #     fig = tonnetz(
    #         df,
    #         colorbar=False,
    #         figsize=(12,12),
    #         cmap='Reds',
    #         # nan_color='white',
    #         edgecolor='black',
    #         show=True,
    #         duration=dur
    #     )
    #     plt.savefig(f'img/pieces/{piece[5:-4]}_tonnetz.png')
    #
    #     # plot inferred distribution
    #     fig = Tone.plot(tones, center, weights=best_weights)
    #     plt.savefig(f'img/pieces/{piece[5:-4]}_estimate.png')
    #     plt.show()
    #
    columns = ['KLs'] + Tone.int_strings + ['diffusion'] + ['piece', 'composer', 'year']
    results = pd.DataFrame(list(zip(KLs, *list(np.array(best_ps).T), pieces, composers, years)), columns=columns)
    results.to_csv(f'results_1.tsv', sep='\t', index=False)
    #
    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(len(JSDs)), JSDs)
    # # plt.xticks(np.arange(len(JSDs)), pieces, rotation=90)
    # ax.plot(JSDs)
    # plt.title("Jensen-Shannon Divergences")
    # plt.tight_layout()
    # plt.show()
