from unittest import TestCase
from diffusion import Tone
import numpy as np


class TestDiffusion(TestCase):
    def test_diffuse(self):
        for open_boundary in [True, False]:
            for min_tone, max_tone in [('F', 'G'), ('Fbb', 'B##')]:
                for _ in range(10):
                    lof = Tone.get_lof(min_tone, max_tone)
                    tones = [Tone((idx, 0), name) for idx, name in enumerate(lof)]
                    Tone.diffuse(tones=tones,
                                 center="C",
                                 atol=1e-2,
                                 action_probs=np.random.uniform(0, 1, 6),
                                 lam=np.random.uniform(0, 1, 1),
                                 open_boundary=open_boundary)
