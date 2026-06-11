from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._plan_metric_sums = {}
		self._plan_metric_counts = {}
		self._eval_reward_auc = 0.
		self._last_eval_step = None
		self._last_eval_reward = None

	@staticmethod
	def _accumulate_metrics(sums, counts, metrics):
		"""Accumulate finite scalars with an independent count per metric."""
		for key, value in metrics.items():
			if torch.is_tensor(value):
				value = value.item()
			value = float(value)
			if not np.isfinite(value):
				continue
			sums[key] = sums.get(key, 0.) + value
			counts[key] = counts.get(key, 0) + 1

	@staticmethod
	def _mean_metrics(sums, counts):
		"""Return per-key means for accumulated scalar metrics."""
		return {
			key: value / counts[key]
			for key, value in sums.items()
			if counts.get(key, 0) > 0
		}

	def _should_log_plan(self, planned_step):
		"""Return whether this planning call should emit full diagnostics."""
		return planned_step % self.cfg.plan_log_freq == 0

	def _record_plan_metrics(self, log_plan=False):
		"""Accumulate planning metrics for episode-level logging."""
		if not self.agent.plan_metrics:
			return
		self._accumulate_metrics(
			self._plan_metric_sums,
			self._plan_metric_counts,
			self.agent.plan_metrics,
		)
		if log_plan:
			self.logger.log_plan(self.agent.plan_metrics, self._step)

	def _consume_plan_metrics(self):
		"""Return episode means and reset planning metric accumulators."""
		if not self._plan_metric_counts:
			return {}
		metrics = self._mean_metrics(
			self._plan_metric_sums,
			self._plan_metric_counts,
		)
		self._plan_metric_sums = {}
		self._plan_metric_counts = {}
		return metrics

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def _environment_metrics(self, info):
		"""Extract scalar environment diagnostics from an episode's final info."""
		metrics = {}
		for key, value in info.items():
			if not key.startswith('metric_'):
				continue
			if torch.is_tensor(value):
				value = value.item()
			metrics[key] = float(value)
		return metrics

	def _update_eval_reward_auc(self, reward):
		"""Update trapezoidal area under the evaluation reward curve."""
		if self._last_eval_step is not None:
			step_delta = self._step - self._last_eval_step
			self._eval_reward_auc += 0.5 * (self._last_eval_reward + reward) * step_delta
		self._last_eval_step = self._step
		self._last_eval_reward = reward
		return self._eval_reward_auc

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths, env_metrics = [], [], [], {}
		plan_metric_sums, plan_metric_counts = {}, {}
		eval_plan_idx = 0
		planned_step = max(self._step - self.cfg.seed_steps - 1, 0)
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				diagnostics = self._should_log_plan(eval_plan_idx)
				action = self.agent.act(
					obs,
					t0=t==0,
					eval_mode=True,
					step=planned_step,
					diagnostics=diagnostics,
				)
				self._accumulate_metrics(
					plan_metric_sums,
					plan_metric_counts,
					self.agent.plan_metrics,
				)
				eval_plan_idx += 1
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			for key, value in self._environment_metrics(info).items():
				env_metrics.setdefault(key, []).append(value)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		metrics = dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)
		metrics.update({key: np.nanmean(values) for key, values in env_metrics.items()})
		metrics.update(self._mean_metrics(plan_metric_sums, plan_metric_counts))
		return metrics

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics['episode_reward_auc'] = self._update_eval_reward_auc(
						eval_metrics['episode_reward']
					)
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self._environment_metrics(info))
					train_metrics.update(self._consume_plan_metrics())
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				explore_step = self._step - self.cfg.seed_steps - 1
				diagnostics = self._should_log_plan(explore_step)
				action = self.agent.act(
					obs,
					t0=len(self._tds)==1,
					step=explore_step,
					diagnostics=diagnostics,
				)
				self._record_plan_metrics(log_plan=diagnostics)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
