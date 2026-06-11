import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_exploration_experiments.py'
SPEC = importlib.util.spec_from_file_location('run_exploration_experiments', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExperimentScriptTest(unittest.TestCase):

	def test_command_does_not_override_save_video(self):
		command = MODULE.command(
			'toy-bimodal',
			'q_bald',
			'triangular',
			1.,
			1,
			30_000,
			20_000,
			'q-bald',
		)
		self.assertFalse(any(arg.startswith('save_video=') for arg in command))


if __name__ == '__main__':
	unittest.main()
