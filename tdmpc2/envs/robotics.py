import gymnasium as gym
import numpy as np

from envs.wrappers.timeout import Timeout


POINT_MAZE_TASKS = {
	'pointmaze-umaze': ('PointMaze_UMaze-v3', 300),
	'pointmaze-open': ('PointMaze_Open-v3', 300),
	'pointmaze-medium': ('PointMaze_Medium-v3', 500),
	'pointmaze-large': ('PointMaze_Large-v3', 700),
}


class PointMazeWrapper(gym.Wrapper):
	"""Flatten goal observations and expose exploration diagnostics."""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		obs_space = env.observation_space.spaces['observation']
		goal_space = env.observation_space.spaces['desired_goal']
		self.observation_space = gym.spaces.Box(
			low=np.concatenate((obs_space.low, goal_space.low)).astype(np.float32),
			high=np.concatenate((obs_space.high, goal_space.high)).astype(np.float32),
			dtype=np.float32,
		)
		self._reset_seed = cfg.seed
		self.action_space.seed(cfg.seed)
		self._free_cells = self._count_free_cells()

	def _count_free_cells(self):
		try:
			maze_map = self.env.unwrapped.maze.maze_map
			return max(sum(cell != 1 for row in maze_map for cell in row), 1)
		except AttributeError:
			return 1

	def _flatten_obs(self, obs):
		return np.concatenate(
			(obs['observation'], obs['desired_goal'])
		).astype(np.float32)

	def _distance(self, obs):
		return float(np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']))

	def _coverage_cell(self, position):
		try:
			cell = self.env.unwrapped.maze.cell_xy_to_rowcol(position)
			return tuple(np.asarray(cell, dtype=np.int64))
		except AttributeError:
			return tuple(np.floor(np.asarray(position) * 2.).astype(np.int64))

	def _update_diagnostics(self, obs, info):
		distance = self._distance(obs)
		self._min_goal_distance = min(self._min_goal_distance, distance)
		self._goal_steps += int(bool(info.get('success', distance <= 0.45)))
		self._success = self._success or self._goal_steps > 0
		self._visited_cells.add(self._coverage_cell(obs['achieved_goal']))
		info.update({
			'success': self._success,
			'metric_goal_reached': float(self._success),
			'metric_goal_steps': float(self._goal_steps),
			'metric_goal_distance': distance,
			'metric_min_goal_distance': self._min_goal_distance,
			'metric_state_coverage': min(
				len(self._visited_cells) / self._free_cells,
				1.,
			),
		})
		return info

	def reset(self):
		obs, info = self.env.reset(seed=self._reset_seed)
		self._reset_seed = None
		self._success = False
		self._goal_steps = 0
		self._min_goal_distance = self._distance(obs)
		self._visited_cells = {self._coverage_cell(obs['achieved_goal'])}
		return self._flatten_obs(obs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		info = self._update_diagnostics(obs, info)
		info['terminated'] = bool(terminated)
		return self._flatten_obs(obs), reward, terminated or truncated, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, **kwargs):
		return self.env.render(**kwargs)


def make_env(cfg):
	"""Make a sparse-reward Gymnasium-Robotics PointMaze environment."""
	if cfg.task not in POINT_MAZE_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'PointMaze tasks only support state observations.'

	try:
		import gymnasium_robotics
	except ImportError as exc:
		raise ImportError(
			'PointMaze requires Gymnasium-Robotics. Install it in the active '
			'environment with: pip install "gymnasium==1.0.0" '
			'"gymnasium-robotics==1.3.0"'
		) from exc

	if hasattr(gym, 'register_envs'):
		gym.register_envs(gymnasium_robotics)
	env_id, episode_length = POINT_MAZE_TASKS[cfg.task]
	env = gym.make(
		env_id,
		render_mode='rgb_array',
		continuing_task=True,
		reset_target=False,
		max_episode_steps=episode_length,
	)
	env = PointMazeWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=episode_length)
	cfg.discount_max = 0.99
	return env
