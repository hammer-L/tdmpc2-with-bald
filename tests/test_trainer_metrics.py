import sys
import unittest
from pathlib import Path


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


if __name__ == '__main__':
	unittest.main()
