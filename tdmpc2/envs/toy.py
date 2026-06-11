import gymnasium as gym
import numpy as np
import torch

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
		height, width = 256, 640
		frame = torch.full((height, width, 3), 245, dtype=torch.uint8)

		def xcoord(position):
			return int(np.clip((position + 1.5) / 3. * (width - 1), 0, width - 1))

		def fill(y0, y1, x0, x1, color):
			x0, x1 = max(0, x0), min(width, x1)
			y0, y1 = max(0, y0), min(height, y1)
			if x1 > x0 and y1 > y0:
				frame[y0:y1, x0:x1] = torch.tensor(color, dtype=torch.uint8)

		# Reward regions and the dynamics-shift region.
		fill(95, 187, xcoord(-0.63), xcoord(-0.07), (191, 220, 246))
		if self.dynamics_shift:
			fill(72, 205, xcoord(0.25), width, (238, 229, 250))
			for stripe in range(xcoord(0.25), width, 24):
				fill(72, 205, stripe, stripe + 5, (224, 207, 241))
		target_color = (53, 190, 108) if self._global_reached else (174, 225, 190)
		fill(84, 198, xcoord(1.16), xcoord(1.24) + 1, target_color)

		# Track and endpoint markers.
		fill(144, 150, 24, width - 24, (72, 76, 82))
		fill(132, 163, 22, 28, (72, 76, 82))
		fill(132, 163, width - 28, width - 22, (72, 76, 82))

		# Episode progress bar.
		fill(24, 36, 32, width - 32, (218, 222, 226))
		progress = min(max(self._t / self._episode_steps, 0.), 1.)
		fill(24, 36, 32, 32 + int((width - 64) * progress), (65, 123, 210))

		# Car body, wheels, and a velocity-direction indicator.
		x = xcoord(self._position)
		car_color = (31, 156, 88) if self._global_reached else (35, 39, 45)
		fill(116, 145, x - 17, x + 18, car_color)
		fill(108, 122, x - 9, x + 10, car_color)
		fill(143, 158, x - 14, x - 3, (18, 20, 23))
		fill(143, 158, x + 4, x + 15, (18, 20, 23))
		velocity_pixels = int(self._velocity / 0.11 * 52)
		if velocity_pixels >= 0:
			fill(102, 107, x, x + velocity_pixels, (225, 83, 70))
			fill(97, 112, x + velocity_pixels - 4, x + velocity_pixels + 2, (225, 83, 70))
		else:
			fill(102, 107, x + velocity_pixels, x, (225, 83, 70))
			fill(97, 112, x + velocity_pixels - 2, x + velocity_pixels + 4, (225, 83, 70))
		return frame.cpu().numpy()


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
