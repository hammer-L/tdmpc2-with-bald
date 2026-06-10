import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	import numpy as np
	from envs.mujoco import make_env
except ImportError:
	np = None
	make_env = None


@unittest.skipIf(make_env is None, 'gymnasium classic-control dependencies are not installed')
class MountainCarEnvTest(unittest.TestCase):

	def test_continuous_mountaincar_adapter(self):
		cfg = SimpleNamespace(
			task='mountaincar-continuous',
			obs='state',
			seed=1,
			discount_max=0.995,
			rho=0.5,
		)
		env = make_env(cfg)
		obs = env.reset()
		self.assertEqual(obs.shape, (2,))
		self.assertEqual(env.action_space.shape, (1,))
		_, _, _, info = env.step(np.array([0.], dtype=np.float32))
		self.assertIn('success', info)
		self.assertIn('terminated', info)
		self.assertIn('metric_max_position', info)


if __name__ == '__main__':
	unittest.main()
