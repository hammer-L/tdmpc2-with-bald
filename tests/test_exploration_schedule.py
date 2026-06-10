import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

from common.exploration import exploration_coefficient, schedule_progress


class ExplorationScheduleTest(unittest.TestCase):

	def coefficient(self, step, schedule='triangular'):
		return exploration_coefficient(
			step=step,
			schedule=schedule,
			peak=2.,
			start=0,
			steps=20_000,
			peak_fraction=0.2,
		)

	def test_triangular_key_points(self):
		self.assertEqual(self.coefficient(0), 0.)
		self.assertAlmostEqual(self.coefficient(4_000), 2.)
		self.assertEqual(self.coefficient(20_000), 0.)
		self.assertEqual(self.coefficient(25_000), 0.)

	def test_triangular_rises_then_falls(self):
		self.assertLess(self.coefficient(1_000), self.coefficient(2_000))
		self.assertGreater(self.coefficient(8_000), self.coefficient(12_000))

	def test_ablation_schedules(self):
		self.assertEqual(self.coefficient(5_000, 'constant'), 2.)
		self.assertAlmostEqual(self.coefficient(10_000, 'linear_decay'), 1.)
		self.assertEqual(self.coefficient(20_000, 'linear_decay'), 0.)

	def test_progress_is_clamped(self):
		self.assertEqual(schedule_progress(-1, 0, 20_000), 0.)
		self.assertEqual(schedule_progress(30_000, 0, 20_000), 1.)


if __name__ == '__main__':
	unittest.main()
