import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	import torch
	from tdmpc2 import TDMPC2
except ImportError:
	torch = None
	TDMPC2 = None


if torch is not None:
	class FakeModel:
		def __init__(self, cfg):
			self.cfg = cfg
			self.next_mc_batch_sizes = []

		def encode(self, obs, task):
			z = torch.zeros(obs.shape[0], self.cfg.latent_dim)
			z[:, :obs.shape[-1]] = obs
			return z

		def pi(self, z, task):
			return torch.zeros(z.shape[0], self.cfg.action_dim), {}

		def next(self, z, action, task):
			return z + 0.01 * action.repeat(1, self.cfg.latent_dim)

		def next_mc(self, z, action, task, samples):
			self.next_mc_batch_sizes.append(z.shape[0])
			predictions = []
			for sample in range(samples):
				logits = z.reshape(z.shape[0], -1, self.cfg.simnorm_dim)
				logits = logits + 0.05 * torch.randn_like(logits)
				logits[..., sample % self.cfg.simnorm_dim] += 0.2
				predictions.append(torch.softmax(logits, dim=-1).reshape_as(z))
			return torch.stack(predictions)

		def reward(self, z, action, task):
			return torch.zeros(z.shape[0], self.cfg.num_bins)

		def Q(self, z, action, task, return_type='avg'):
			if return_type == 'all':
				bins = torch.linspace(-1., 1., self.cfg.num_bins)
				return torch.stack([
					bins.repeat(z.shape[0], 1) * (head + 1) * 0.1
					for head in range(self.cfg.num_q)
				])
			return torch.zeros(z.shape[0], 1)


@unittest.skipIf(torch is None, 'PyTorch/tensordict are not installed')
class ExplorationSmokeTest(unittest.TestCase):

	def make_agent(self, mode):
		cfg = SimpleNamespace(
			explore_reward=mode,
			q_bald_num_q=5,
			dynamics_bald_samples=5,
			explore_noise_std=1.,
			latent_dim=8,
			simnorm_dim=4,
			action_dim=1,
			num_q=5,
			num_bins=11,
			vmin=-2.,
			vmax=2.,
			bin_size=0.4,
			num_samples=16,
			num_elites=4,
			num_pi_trajs=0,
			horizon=3,
			iterations=2,
			max_std=1.,
			min_std=0.05,
			temperature=0.5,
			multitask=False,
			episodic=False,
			mpc=True,
			explore_schedule='triangular',
			explore_coef_peak=1.,
			explore_schedule_start=0,
			explore_schedule_steps=20_000,
			explore_peak_fraction=0.2,
			bald_diagnostics=True,
			plan_log_freq=10,
			plan_alignment_target=0.2,
		)
		agent = SimpleNamespace(
			cfg=cfg,
			model=FakeModel(cfg),
			device=torch.device('cpu'),
			discount=0.99,
			_prev_mean=torch.zeros(cfg.horizon, cfg.action_dim),
		)
		agent._exploration_reward = types.MethodType(TDMPC2._exploration_reward, agent)
		agent._bald_reward = types.MethodType(TDMPC2._bald_reward, agent)
		agent._estimate_value = types.MethodType(TDMPC2._estimate_value, agent)
		agent._plan_diagnostics = types.MethodType(TDMPC2._plan_diagnostics, agent)
		agent._explore_coefficient = types.MethodType(TDMPC2._explore_coefficient, agent)
		agent.plan = types.MethodType(TDMPC2._plan, agent)
		agent._last_plan_metrics = {}
		return agent

	def test_all_exploration_modes_produce_finite_mppi_outputs(self):
		obs = torch.zeros(1, 3)
		for mode in ('none', 'q_bald', 'dynamics_bald', 'noise'):
			with self.subTest(mode=mode):
				agent = self.make_agent(mode)
				action, stats, elite_data = TDMPC2._plan(
					agent,
					obs,
					t0=True,
					eval_mode=True,
					task=None,
					explore=mode != 'none',
					explore_coef=torch.tensor(0.5),
				)
				self.assertTrue(torch.isfinite(action).all())
				self.assertTrue(torch.isfinite(stats).all())
				self.assertEqual(elite_data[0].shape[1], agent.cfg.num_elites)

	def test_evaluation_disables_exploration(self):
		agent = self.make_agent('q_bald')
		call = {}

		def plan(obs, **kwargs):
			call.update(kwargs)
			elite_shape = (agent.cfg.horizon, agent.cfg.num_elites, agent.cfg.action_dim)
			return (
				torch.zeros(1),
				torch.zeros(8),
				(
					torch.zeros(elite_shape),
					torch.zeros(agent.cfg.num_elites, 1),
					torch.zeros(agent.cfg.num_elites, 1),
					torch.zeros(agent.cfg.num_elites, 1),
				),
			)

		agent.plan = plan
		action = TDMPC2.act(
			agent,
			torch.zeros(3),
			eval_mode=True,
			step=4_000,
		)
		self.assertTrue(torch.isfinite(action).all())
		self.assertFalse(call['explore'])
		self.assertEqual(call['explore_coef'].item(), 0.)
		self.assertEqual(agent._last_plan_metrics['explore_coefficient'], 0.)
		self.assertIn('planning_task_value_mean', agent._last_plan_metrics)

	def test_none_mode_reports_both_bald_diagnostics(self):
		agent = self.make_agent('none')
		action = TDMPC2.act(
			agent,
			torch.zeros(3),
			eval_mode=False,
			step=0,
			diagnostics=True,
		)
		self.assertTrue(torch.isfinite(action).all())
		for key in (
			'elite_task_return_mean',
			'q_bald_mean',
			'q_bald_return_mean',
			'dynamics_bald_mean',
			'dynamics_bald_return_mean',
		):
			self.assertIn(key, agent._last_plan_metrics)
			self.assertTrue(np.isfinite(agent._last_plan_metrics[key]))
		self.assertTrue(all(
			size == agent.cfg.num_elites
			for size in agent.model.next_mc_batch_sizes
		))

	def test_disabled_bald_diagnostics_still_reports_elite_return(self):
		agent = self.make_agent('none')
		agent.cfg.bald_diagnostics = False
		TDMPC2.act(
			agent,
			torch.zeros(3),
			eval_mode=False,
			step=0,
			diagnostics=True,
		)
		self.assertIn('elite_task_return_mean', agent._last_plan_metrics)
		self.assertNotIn('q_bald_mean', agent._last_plan_metrics)
		self.assertEqual(agent.model.next_mc_batch_sizes, [])

	def test_evaluation_reports_diagnostics_with_zero_coefficient(self):
		agent = self.make_agent('q_bald')
		TDMPC2.act(
			agent,
			torch.zeros(3),
			eval_mode=True,
			step=4_000,
			diagnostics=True,
		)
		self.assertEqual(agent._last_plan_metrics['explore_coefficient'], 0.)
		self.assertEqual(
			agent._last_plan_metrics['active_explore_bonus_task_ratio'],
			0.,
		)
		self.assertIn('q_bald_mean', agent._last_plan_metrics)
		self.assertIn('dynamics_bald_mean', agent._last_plan_metrics)

	def test_diagnostics_preserve_action_and_rng_state(self):
		agent = self.make_agent('none')
		obs = torch.zeros(3)

		torch.manual_seed(7)
		agent._prev_mean.zero_()
		action_without = TDMPC2.act(
			agent, obs, t0=True, step=0, diagnostics=False
		)
		next_without = torch.rand(4)

		torch.manual_seed(7)
		agent._prev_mean.zero_()
		action_with = TDMPC2.act(
			agent, obs, t0=True, step=0, diagnostics=True
		)
		next_with = torch.rand(4)

		self.assertTrue(torch.equal(action_without, action_with))
		self.assertTrue(torch.equal(next_without, next_with))

	def test_alignment_coefficient_uses_elite_task_return(self):
		agent = self.make_agent('none')
		elite_actions = torch.zeros(
			agent.cfg.horizon,
			agent.cfg.num_elites,
			agent.cfg.action_dim,
		)
		model_return = torch.ones(agent.cfg.num_elites, 1)
		terminal_q = torch.ones(agent.cfg.num_elites, 1)
		active_return = torch.zeros(agent.cfg.num_elites, 1)
		metrics = TDMPC2._plan_diagnostics(
			agent,
			torch.zeros(1, 3),
			(elite_actions, model_return, terminal_q, active_return),
			None,
			0.,
		)
		self.assertAlmostEqual(metrics['elite_task_return_mean'], 2.)
		self.assertAlmostEqual(
			metrics['suggested_q_bald_coefficient'],
			agent.cfg.plan_alignment_target / metrics['q_bald_unit_ratio'],
			places=6,
		)


if __name__ == '__main__':
	unittest.main()
