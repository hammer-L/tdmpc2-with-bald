import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	import gymnasium as gym
	import numpy as np
	from envs.robotics import PointMazeWrapper, make_env
except ImportError:
	gym = None
	np = None
	PointMazeWrapper = None
	make_env = None


if gym is not None:
	class FakeMaze:
		maze_map = [
			[1, 1, 1, 1],
			[1, 0, 0, 1],
			[1, 0, 0, 1],
			[1, 1, 1, 1],
		]

		def cell_xy_to_rowcol(self, position):
			return np.floor(position).astype(np.int64)


	class FakePointMaze(gym.Env):
		def __init__(self):
			self.observation_space = gym.spaces.Dict({
				'observation': gym.spaces.Box(-np.inf, np.inf, (4,), np.float64),
				'achieved_goal': gym.spaces.Box(-np.inf, np.inf, (2,), np.float64),
				'desired_goal': gym.spaces.Box(-np.inf, np.inf, (2,), np.float64),
			})
			self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
			self.maze = FakeMaze()
			self._position = np.zeros(2)

		def _obs(self):
			return {
				'observation': np.concatenate((self._position, np.zeros(2))),
				'achieved_goal': self._position.copy(),
				'desired_goal': np.ones(2),
			}

		def reset(self, *, seed=None, options=None):
			self._position = np.zeros(2)
			return self._obs(), {}

		def step(self, action):
			self._position = self._position + np.asarray(action)
			success = np.linalg.norm(self._position - 1.) <= 0.45
			return self._obs(), float(success), False, False, {'success': success}


@unittest.skipIf(PointMazeWrapper is None, 'gymnasium is not installed')
class PointMazeWrapperTest(unittest.TestCase):

	def test_flattens_goal_observation(self):
		env = PointMazeWrapper(
			FakePointMaze(),
			SimpleNamespace(seed=3),
		)
		obs = env.reset()
		self.assertEqual(obs.shape, (6,))
		self.assertEqual(obs.dtype, np.float32)
		self.assertEqual(env.action_space.shape, (2,))
		np.testing.assert_allclose(obs[-2:], np.ones(2))

	def test_tracks_success_distance_and_coverage(self):
		env = PointMazeWrapper(
			FakePointMaze(),
			SimpleNamespace(seed=3),
		)
		env.reset()
		_, reward, done, info = env.step(
			np.array([1., 1.], dtype=np.float32)
		)
		self.assertEqual(reward, 1.)
		self.assertFalse(done)
		self.assertTrue(info['success'])
		self.assertFalse(info['terminated'])
		self.assertEqual(info['metric_goal_reached'], 1.)
		self.assertEqual(info['metric_goal_distance'], 0.)
		self.assertGreater(info['metric_state_coverage'], 0.)


try:
	import gymnasium_robotics
except ImportError:
	gymnasium_robotics = None


@unittest.skipIf(
	make_env is None or gymnasium_robotics is None,
	'gymnasium-robotics is not installed',
)
class PointMazeIntegrationTest(unittest.TestCase):

	def test_umaze_adapter(self):
		cfg = SimpleNamespace(
			task='pointmaze-umaze',
			obs='state',
			seed=1,
			discount_max=0.995,
		)
		env = make_env(cfg)
		obs = env.reset()
		self.assertEqual(obs.shape, (6,))
		self.assertEqual(env.action_space.shape, (2,))
		_, _, _, info = env.step(
			np.zeros(2, dtype=np.float32)
		)
		self.assertIn('metric_min_goal_distance', info)
		self.assertIn('metric_state_coverage', info)
		env.close()


if __name__ == '__main__':
	unittest.main()
