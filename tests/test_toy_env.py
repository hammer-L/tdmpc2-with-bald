import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	import numpy as np
	from envs.toy import BimodalEnv, make_env
except ImportError:
	np = None
	BimodalEnv = None
	make_env = None


@unittest.skipIf(BimodalEnv is None, 'gymnasium is not installed')
class BimodalEnvTest(unittest.TestCase):

	def test_spaces_and_timeout(self):
		cfg = SimpleNamespace(task='toy-bimodal', obs='state', seed=1, discount_max=0.995)
		env = make_env(cfg)
		obs = env.reset()
		self.assertEqual(obs.shape, (3,))
		self.assertEqual(env.action_space.shape, (1,))
		for step in range(50):
			obs, reward, done, info = env.step(np.array([0.], dtype=np.float32))
			if step < 49:
				self.assertFalse(done)
		self.assertTrue(done)
		self.assertEqual(step + 1, 50)
		self.assertFalse(info['terminated'])

	def test_seed_reproducibility(self):
		env_a = BimodalEnv(seed=7)
		env_b = BimodalEnv(seed=7)
		np.testing.assert_allclose(env_a.reset(seed=7), env_b.reset(seed=7))
		for action in (-1., 0.3, 1., -0.2):
			out_a = env_a.step(np.array([action], dtype=np.float32))
			out_b = env_b.step(np.array([action], dtype=np.float32))
			np.testing.assert_allclose(out_a[0], out_b[0])
			self.assertEqual(out_a[1:], out_b[1:])

	def test_final_info_metrics(self):
		env = BimodalEnv(dynamics_shift=True, seed=3)
		env.reset(seed=3)
		info = None
		for _ in range(50):
			_, _, _, info = env.step(np.array([0.4], dtype=np.float32))
		expected = {
			'success',
			'metric_global_reached',
			'metric_global_steps',
			'metric_local_steps',
			'metric_max_position',
			'metric_state_coverage',
		}
		self.assertTrue(expected.issubset(info))
		self.assertGreaterEqual(info['metric_state_coverage'], 0.)
		self.assertLessEqual(info['metric_state_coverage'], 1.)

	def test_global_reward_is_reachable_in_both_variants(self):
		base = BimodalEnv(seed=1)
		base.reset(seed=1)
		for _ in range(50):
			_, _, _, base_info = base.step(np.array([1.], dtype=np.float32))
		self.assertTrue(base_info['success'])

		shifted = BimodalEnv(dynamics_shift=True, seed=1)
		shifted.reset(seed=1)
		for _ in range(50):
			_, _, _, shifted_info = shifted.step(np.array([1/3], dtype=np.float32))
		self.assertTrue(shifted_info['success'])

		shifted.reset(seed=1)
		for _ in range(50):
			_, _, _, naive_info = shifted.step(np.array([1.], dtype=np.float32))
		self.assertFalse(naive_info['success'])

	def test_render_shape(self):
		env = BimodalEnv(seed=1)
		self.assertEqual(env.render().shape, (96, 384, 3))


if __name__ == '__main__':
	unittest.main()
