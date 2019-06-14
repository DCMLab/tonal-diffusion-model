import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

### Functions to generate arbitrary line-of-fifths (lof) segment
steps = {s:idx for s, idx in zip(list('FCGDAEB'), range(7))}

def get_lof_no(tpc):
    """Returns tpc from lof number"""
    step = steps[tpc[0]]
    accs = tpc[1:].count('#') - tpc[1:].count('b')

    return step + 7 * accs

def get_tpc(lof_no):
    """Returns tpc for lof number"""
    a, b = divmod(lof_no, 7)
    d = {v:k for k, v in steps.items()}
    tpc = d[b]
    if a < 0:
        tpc += abs(a) * 'b'
    if a > 0:
        tpc += abs(a) * '#'
    return tpc

def get_lof(min_tpc, max_tpc):
    """Returns number range on from tpcs"""
    min = get_lof_no(min_tpc)
    max = get_lof_no(max_tpc)

    if max < min:
        raise UserWarning(f"{min_tpc} is not lower than {max_tpc}.")
    return [get_tpc(l) for l in np.arange(min, max + 1)]

class Tone:

    e_x = np.array([1, 0])
    e_y = np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)])
    eq = np.array([3, 1])

    @staticmethod
    def plot(tones, weights, min_col=(1, 1, 1), max_col=(1, 0, 0)):
        weights = np.array(weights)
        weights /= np.max(weights)
        min_col = np.array(min_col)
        max_col = np.array(max_col)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        center_x, center_y = [t.loc for t in tones if t.name == 'C'][0]
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

    @staticmethod
    def diffuse(tones, action_probs, discount=0.9, init_dist=None, max_iter=1000):
        n = len(tones)
        # initialize init_dist if not provided or convert to numpy array
        if init_dist is None:
            init_dist = np.zeros(n)
            for idx, t in enumerate(tones):
                if t.name == 'C':# or t.name == 'G':
                    init_dist[idx] = 1 #0.5
        else:
            init_dist = np.array(init_dist)
        if not np.isclose(np.sum(init_dist), 1):
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
    lof = get_lof('Fbb', 'B##')
    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]
    weights = Tone.diffuse(tones=tones, action_probs=[5, 1, 1, 3, 3, 1], discount=.5)
    Tone.plot(tones, weights=weights)
