import matplotlib as mpl
import matplotlib.pyplot as plt
from pitchplots.static import tonnetz

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import pandas as pd

import glob

class Tone:

    ### Functions to generate arbitrary line-of-fifths (lof) segment
    steps = {s:idx for s, idx in zip(list('FCGDAEB'), range(7))}

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

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
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
                        verticalalignment='center')
                ax.add_patch(mpl.patches.CirclePolygon(
                    loc,
                    radius=np.tan(np.pi / 6),
                    resolution=6,
                    facecolor=cmap(w),
                    edgecolor=(0, 0, 0)))
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    ### Diffusion

    intervals = [
        1,   # fifth up
        -1,  # fifth down
        -3,  # minor third up
        3,   # minor third down
        4,  # major third up
        -4    # major third down
    ]

    @staticmethod
    def diffuse(tones, center, action_probs, discount=[0.5]*6, init_dist=None, max_iter=1000):
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

        # normalize action_probs and convert to numpy array
        action_probs = np.array(action_probs)
        action_probs = action_probs / np.sum(action_probs)
        # initialize transition matrices
        transition_matrices = [np.zeros((n, n)) for _ in action_probs]
        for from_idx in range(n):
            for to_idx in range(n):
                for mat_idx, step in enumerate(Tone.intervals):
                    if to_idx - from_idx == step:
                        transition_matrices[mat_idx][from_idx, to_idx] = discount[mat_idx]

        # diffuse
        current_dist = init_dist.copy()
        next_dist = np.zeros_like(current_dist)
        for iteration in range(max_iter):
            np.copyto(next_dist, init_dist)
            for a_idx, a_prob in enumerate(action_probs):
                next_dist += a_prob * np.einsum('i,ij->j', current_dist, transition_matrices[a_idx])
            if np.all(np.isclose(current_dist, next_dist, atol=1e-2)):
                break
            else:
                # use new distribution as current for next iteration
                np.copyto(current_dist, next_dist)
        else:
            raise UserWarning(f"Did not converge after {iteration} iterations")
        return next_dist

    def __init__(self, loc, name, weight=0):
        self.loc = np.array(loc)
        self.name = name

    def get_loc(self, eq_idx=0):
        x, y = self.loc + Tone.eq * eq_idx
        return x * Tone.e_x + y * Tone.e_y

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
        df = pd.read_csv(csv_path, index_col=0)
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
        freqs = freqs.reindex(lof).fillna(0).values
        center = df.tpc.value_counts().idxmax()
        return freqs, center


if __name__ == "__main__":
    lof = Tone.get_lof('Fbb', 'B##')
    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]

    ### Example pieces
    pieces = [
        'data/machaut_detoutes.csv',
        'data/Gesualdo_OVos.csv',
        'data/Salve-Regina_Lasso.csv',
        'data/BWV_846.csv',
        'data/sonata01-1.csv',
        'data/Sonate_No._17_Tempest_1st_Movement.csv',
        'data/Schubert_90_2.csv',
        'data/Wanderer_Fantasy.csv',
        'data/Chopin_Opus_28_4.csv',
        'data/Liszt_Lugubre_gondola_I_200_1.csv',
        'data/Brahms_116_1.csv',
        'data/Satie_-_Gnossiennes_1.csv',
        'data/Ravel_-_Miroirs_I.csv',
        'data/Webern_Variationen_1.csv'
    ]

    meta = pd.read_csv('../ExtendedTonality/metadata.csv', sep='\t', encoding='utf-8')
    meta = meta[meta.filename.notnull()]
    path = '..\\ExtendedTonality\\data\\DataFrames\\'
    csvs = [f for f in glob.glob(path+"*.csv")][::100]
    pieces = []
    composers = []
    years = []
    for i, row in meta.iterrows():
        if  (path + row.filename + '.csv' in csvs):
            pieces.append(path + row.filename + '.csv')
            composers.append(row.composer)
            years.append(row.display_year)
    ### INFERENCE
    # Constraint 1: weights and discounts must be between 0 and 1
    bnds = ((0, 1),) * 6 * 2 # 6 step directions plus discount
    # Constraint 2: sum of weights must be 1
    def con(a):
        return sum(a[:6]) - 1
    cons = {'type':'eq', 'fun': con}

    def cost_f(x, args):
        weights = Tone.diffuse(tones=tones, center=center, action_probs=x[:-6], discount=x[-6:])
        weights /= weights.sum()
        return Tone.jsd(weights, args)

    JSDs = []
    best_ws = []
    for piece in pieces:
        freqs, center = Tone.piece_freqs(piece, by_duration=True)

        mini = minimize(
            fun=cost_f,
            x0=[1/6] * 6 * 2,
            args=(freqs),
            method="SLSQP", # Sequential Least SQares Programming
            bounds=bnds,
            constraints=cons
        )
        best_params = mini.get('x')
        best_weights = Tone.diffuse(tones=tones, center=center, action_probs=best_params[:-6], discount=best_params[-6:])
        best_weights /= best_weights.sum()

        JSDs.append(Tone.jsd(freqs, best_weights))
        best_ws.append(best_weights.T)


        # results = pd.DataFrame(list(zip(JSDs, *best_ws, pieces, composers)))
        # results.to_csv('results.tsv', sep='\t', index=False)

        ### PLOT
        # # plot optimal parameters
        # x = np.arange(best_params[:-6].shape[0])
        # plt.bar(x, best_params[:-6])
        # ds = [round(p,3) for p in best_params[-6:]]
        # plt.xticks(x, [f'+P5\n{ds[0]}', f'-P5\n{ds[1]}', f'+m3\n{ds[2]}', f'-m3\n{ds[3]}', f'+M3\n{ds[4]}', f'-M3\n{ds[5]}'])
        # plt.show()
        #
        # # plot both distributions
        # pd.DataFrame(
        #     {'original':freqs, 'estimate':best_weights}
        #     ).plot(
        #         kind='bar',
        #         figsize=(12,6)
        #     )
        # plt.title(f"JSD: {round(Tone.jsd(freqs, best_weights), 3)}")
        # plt.xticks(np.arange(len(lof)),lof)
        # plt.tight_layout()
        # plt.show()
        #
        # # plot actual distribution
        # df =pd.read_csv(piece)
        # df['tpc'] = df['tpc'].str.replace('x', '##')
        # fig = tonnetz(
        #     df,
        #     colorbar=False,
        #     figsize=(12,12),
        #     cmap='Reds',
        #     # nan_color='white',
        #     edgecolor='black',
        #     show=True
        # )
        #
        # # plot inferred distribution
        # Tone.plot(tones, center, weights=best_weights)

    fig, ax = plt.subplots(figsize=(18,15))
    # ax.scatter(np.arange(len(JSDs)), JSDs)
    # plt.xticks(np.arange(len(JSDs)), pieces, rotation=90)
    ax.plot(JSDs)
    plt.title("Jensen-Shannon Divergences")
    plt.tight_layout()
    plt.show()
