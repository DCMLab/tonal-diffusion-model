from unittest import TestCase
from diffusion import Tone
import numpy as np


class TestDiffusion(TestCase):
    def test_diffuse(self):
        # check boundary flag
        for open_boundary in [True, False]:
            # check for small and large tonal ranges
            for min_tone, max_tone in [('F', 'G'), ('Fbb', 'B##')]:
                # check multiple times randomized
                for _ in range(10):
                    lof = Tone.get_lof(min_tone, max_tone)
                    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]
                    # check against old version
                    action_probs = np.random.uniform(0, 1, 6)
                    lam = np.random.uniform(0, 1, 1)[0]
                    weights_ = Tone.diffuse_(tones=tones,
                                             center="C",
                                             atol=1e-10,
                                             action_probs=action_probs,
                                             lam=lam,
                                             open_boundary=open_boundary,
                                             norm_lambda=True)
                    weights = Tone.diffuse(tones=tones,
                                           center="C",
                                           atol=1e-10,
                                           action_probs=action_probs,
                                           lam=lam,
                                           open_boundary=open_boundary)
                    np.testing.assert_array_almost_equal(weights, weights_, decimal=4)
