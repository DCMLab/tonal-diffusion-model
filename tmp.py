import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import pandas as pd

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
    def plot(tones, center, weights, min_col=(1, 1, 1), max_col=(1, 0, 0)):
        weights = np.array(weights)
        weights /= np.max(weights)
        min_col = np.array(min_col)
        max_col = np.array(max_col)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        center_x, center_y = [t.loc for t in tones if t.name == center][0]
        x_min = center_x - 5
        x_max = center_x + 5
        y_min = center_y - 5
        y_max = center_y + 5
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
                    facecolor=w * max_col + (1 - w) * min_col,
                    edgecolor=(0, 0, 0)))
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    ### Diffusion
    @staticmethod
    def diffuse(tones, center, action_probs, discount=0.9, init_dist=None, max_iter=10_000):
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
                for mat_idx, step in enumerate([
                    1,   # fifth up
                    -1,  # fifth down
                    -3,  # minor third up
                    3,   # minor third down
                    4,  # major third up
                    -4    # major third down
                ]):
                    if to_idx - from_idx == step:
                        transition_matrices[mat_idx][from_idx, to_idx] = discount

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


if __name__ == "__main__":
    lof = Tone.get_lof('Fbb', 'B##')
    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]

    # path = 'data/Satie_-_Gnossiennes_1.csv'
    path = 'data/BWV_772.csv'
    piece = pd.read_csv(path)
    counts = piece.tpc.value_counts(normalize=True).reindex(lof).fillna(0).values
    center = piece.tpc.value_counts().idxmax()

    ### INFERENCE
    # Constraint 1: probs must be between 0 and 1
    bnds = ((0, 1),) * 7 # 6 step directions plus discount

    # Constraint 2: sum must be 1
    def con(a):
        return sum(a[:6]) - 1

    cons = {'type':'eq', 'fun': con}

    def jsd(p, q, base=2):
        ## convert to np.array
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        p, q = p/p.sum(), q/q.sum()
        m = 1./2*(p + q)
        return entropy(p,m, base=base)/2. + entropy(q, m, base=base)/2.

    def cost_f(x, args):
        weights = Tone.diffuse(tones=tones, center=center, action_probs=x, discount=.99)
        weights /= weights.sum()

        return jsd(weights, args)

    mini = minimize(fun=cost_f, x0=[1/6]*6+[.99], args=(counts), method="SLSQP", bounds=bnds, constraints=cons)
    best_params = mini.get('x')
    best_weights = Tone.diffuse(tones=tones, center=center, action_probs=best_params[:-1], discount=best_params[-1])
    print(best_params)
    ### PLOT
    Tone.plot(tones, center, weights=best_weights)
