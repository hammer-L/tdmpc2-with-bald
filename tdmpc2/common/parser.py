import dataclasses
import re
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

	# Multi-task
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		# Account for slight inconsistency in task_dim for the mt30 experiments
		cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

	# Planning-only exploration
	valid_explore_rewards = {'none', 'q_bald', 'dynamics_bald', 'noise'}
	assert cfg.explore_reward in valid_explore_rewards, \
		f'Invalid explore_reward {cfg.explore_reward}. Must be one of {sorted(valid_explore_rewards)}'
	assert cfg.explore_schedule in {'triangular', 'constant', 'linear_decay'}, \
		'Explore schedule must be one of [triangular, constant, linear_decay].'
	assert cfg.explore_schedule_start >= 0, 'explore_schedule_start must be non-negative.'
	assert cfg.explore_schedule_steps > 0, 'explore_schedule_steps must be positive.'
	assert cfg.explore_coef_peak >= 0, 'explore_coef_peak must be non-negative.'
	assert 0 < cfg.explore_peak_fraction < 1, 'explore_peak_fraction must be in (0, 1).'
	assert cfg.q_bald_num_q > 1, 'Q-BALD requires at least two Q heads.'
	assert 0 <= cfg.dynamics_dropout < 1, 'dynamics_dropout must be in [0, 1).'
	assert cfg.explore_noise_std >= 0, 'explore_noise_std must be non-negative.'
	if cfg.explore_reward == 'q_bald':
		assert cfg.num_bins > 1, 'Q-BALD requires categorical Q outputs (num_bins > 1).'
		assert cfg.num_q >= cfg.q_bald_num_q, \
			f'Q-BALD requested {cfg.q_bald_num_q} Q heads, but the model only has {cfg.num_q}.'
	if cfg.explore_reward == 'dynamics_bald':
		assert cfg.dynamics_dropout > 0, 'Dynamics-BALD requires dynamics_dropout > 0.'
		assert cfg.dynamics_bald_samples > 1, 'Dynamics-BALD requires at least two MC samples.'
		assert cfg.latent_dim % cfg.simnorm_dim == 0, \
			'latent_dim must be divisible by simnorm_dim for Dynamics-BALD.'

	return cfg_to_dataclass(cfg)
