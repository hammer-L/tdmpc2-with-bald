import gymnasium as gym
import numpy as np

from envs.wrappers.timeout import Timeout


TOY_TASKS = {'toy-bimodal', 'toy-bimodal-dynamics'}


class BimodalEnv(gym.Env):
	"""
	Continuous-control diagnostic with an easy local reward and a narrow global reward.
	The dynamics variant changes action response only in the initially unvisited right region.
	"""

	metadata = {'render_modes': ['rgb_array'], 'render_fps': 15}

	def __init__(self, dynamics_shift=False, seed=0):
		super().__init__()
		self.dynamics_shift = dynamics_shift
		self.render_mode = 'rgb_array'
		self.observation_space = gym.spaces.Box(
			low=np.array([-1.5, -0.12, 0.], dtype=np.float32),
			high=np.array([1.5, 0.12, 1.], dtype=np.float32),
			dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=-1., high=1., shape=(1,), dtype=np.float32
		)
		self.action_space.seed(seed)
		self._rng = np.random.default_rng(seed)
		self._episode_steps = 50
		self.reset(seed=seed)

	def _obs(self):
		return np.array([
			self._position,
			self._velocity,
			self._t / self._episode_steps,
		], dtype=np.float32)

	def _reward(self):
		local = 0.25 * np.exp(-0.5 * ((self._position + 0.35) / 0.28) ** 2)
		global_reward = np.exp(-0.5 * ((self._position - 1.2) / 0.04) ** 2)
		return float(max(local, global_reward))

	def _info(self):
		return {
			'success': self._global_reached,
			'terminated': False,
			'metric_global_reached': float(self._global_reached),
			'metric_global_steps': float(self._global_steps),
			'metric_local_steps': float(self._local_steps),
			'metric_max_position': float(self._max_position),
			'metric_state_coverage': len(self._visited_bins) / 24.,
		}

	def reset(self, *, seed=None, options=None):
		super().reset(seed=seed)
		if seed is not None:
			self._rng = np.random.default_rng(seed)
			self.action_space.seed(seed)
		self._position = float(self._rng.uniform(-0.02, 0.02))
		self._velocity = 0.
		self._t = 0
		self._global_reached = False
		self._global_steps = 0
		self._local_steps = 0
		self._max_position = self._position
		self._visited_bins = {self._coverage_bin(self._position)}
		return self._obs()

	def _coverage_bin(self, position):
		return int(np.clip((position + 1.5) / 3. * 24, 0, 23))

	def step(self, action):
		action = float(np.clip(np.asarray(action).reshape(-1)[0], -1., 1.))
		if self.dynamics_shift and self._position > 0.25:
			force = 0.075 * np.exp(-0.5 * ((action - 0.33) / 0.14) ** 2) - 0.028
		else:
			force = 0.045 * action
		self._velocity = float(np.clip(
			0.85 * self._velocity + force - 0.006 * self._position,
			-0.11,
			0.11,
		))
		self._position = float(np.clip(
			self._position + self._velocity,
			-1.5,
			1.5,
		))
		self._t += 1

		in_local = abs(self._position + 0.35) < 0.28
		in_global = abs(self._position - 1.2) < 0.04
		self._local_steps += int(in_local)
		self._global_steps += int(in_global)
		self._global_reached = self._global_reached or in_global
		self._max_position = max(self._max_position, self._position)
		self._visited_bins.add(self._coverage_bin(self._position))
		return self._obs(), self._reward(), False, self._info()

	def render(self):
		height, width = 96, 384
		frame = np.full((height, width, 3), 245, dtype=np.uint8)

		def xcoord(position):
			return int(np.clip((position + 1.5) / 3. * (width - 1), 0, width - 1))

		frame[42:55, xcoord(-0.63):xcoord(-0.07)] = (190, 220, 245)
		frame[37:60, xcoord(1.16):xcoord(1.24)] = (180, 230, 190)
		frame[47:50, :] = (80, 80, 80)
		x = xcoord(self._position)
		frame[34:63, max(0, x-4):min(width, x+5)] = (25, 25, 25)
		return frame


def make_env(cfg):
	"""Make a lightweight TD-MPC2 exploration diagnostic."""
	if cfg.task not in TOY_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'Toy tasks only support state observations.'
	env = BimodalEnv(
		dynamics_shift=cfg.task == 'toy-bimodal-dynamics',
		seed=cfg.seed,
	)
	cfg.discount_max = 0.99
	return Timeout(env, max_episode_steps=50)
