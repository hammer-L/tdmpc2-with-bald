import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout


MUJOCO_TASKS = {
	'mujoco-walker': 'Walker2d-v4',
	'mujoco-halfcheetah': 'HalfCheetah-v4',
	'bipedal-walker': 'BipedalWalker-v3',
	'lunarlander-continuous': 'LunarLander-v2',
	'mountaincar-continuous': 'MountainCarContinuous-v0',
}

class MuJoCoWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._cumulative_reward = 0
		self._max_position = -np.inf
		self._reset_seed = cfg.seed
		self.action_space.seed(cfg.seed)

	def reset(self):
		self._cumulative_reward = 0
		obs = self.env.reset(seed=self._reset_seed)[0]
		self._reset_seed = None
		self._max_position = float(obs[0]) if self.cfg.task == 'mountaincar-continuous' else -np.inf
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		self._cumulative_reward += reward
		done = terminated or truncated
		info['terminated'] = terminated
		if self.cfg.task == 'lunarlander-continuous':
			info['success'] = self._cumulative_reward > 200
		elif self.cfg.task == 'mountaincar-continuous':
			self._max_position = max(self._max_position, float(obs[0]))
			info['success'] = bool(terminated)
			info['metric_max_position'] = self._max_position
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		return self.env.render(**kwargs)


def make_env(cfg):
	"""
	Make classic/MuJoCo environment.
	"""
	if not cfg.task in MUJOCO_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	if cfg.task == 'lunarlander-continuous':
		env = gym.make(MUJOCO_TASKS[cfg.task], continuous=True, render_mode='rgb_array')
	else:
		env = gym.make(MUJOCO_TASKS[cfg.task], render_mode='rgb_array')
	env = MuJoCoWrapper(env, cfg)
	env = Timeout(env, max_episode_steps={
		'lunarlander-continuous': 500,
		'bipedal-walker': 1600,
		'mountaincar-continuous': 999,
	}.get(cfg.task, 1000)) # Default max episode steps for other tasks
	cfg.discount_max = 0.99 # TODO: temporarily hardcode for these envs, makes comparison to other codebases easier
	cfg.rho = 0.7 # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
	return env
