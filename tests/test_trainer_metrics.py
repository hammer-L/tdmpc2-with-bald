import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	from trainer.online_trainer import OnlineTrainer
except ImportError:
	OnlineTrainer = None


@unittest.skipIf(OnlineTrainer is None, 'PyTorch/tensordict are not installed')
class TrainerMetricsTest(unittest.TestCase):

	def test_eval_reward_auc(self):
		trainer = object.__new__(OnlineTrainer)
		trainer._eval_reward_auc = 0.
		trainer._last_eval_step = None
		trainer._last_eval_reward = None

		trainer._step = 0
		self.assertEqual(trainer._update_eval_reward_auc(1.), 0.)
		trainer._step = 10
		self.assertEqual(trainer._update_eval_reward_auc(3.), 20.)
		trainer._step = 20
		self.assertEqual(trainer._update_eval_reward_auc(5.), 60.)

	def test_plan_metrics_use_independent_counts(self):
		sums, counts = {}, {}
		OnlineTrainer._accumulate_metrics(
			sums,
			counts,
			{'planning_task_value_mean': 2., 'q_bald_mean': 1.},
		)
		OnlineTrainer._accumulate_metrics(
			sums,
			counts,
			{'planning_task_value_mean': 4.},
		)
		metrics = OnlineTrainer._mean_metrics(sums, counts)
		self.assertEqual(metrics['planning_task_value_mean'], 3.)
		self.assertEqual(metrics['q_bald_mean'], 1.)

	def test_plan_log_frequency(self):
		trainer = object.__new__(OnlineTrainer)
		trainer.cfg = SimpleNamespace(plan_log_freq=10)
		for step in (0, 10, 20):
			self.assertTrue(trainer._should_log_plan(step))
		for step in (1, 9, 11, 19):
			self.assertFalse(trainer._should_log_plan(step))


if __name__ == '__main__':
	unittest.main()
