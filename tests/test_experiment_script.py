import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_exploration_experiments.py'
SPEC = importlib.util.spec_from_file_location('run_exploration_experiments', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SEARCH_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'search_exploration_coef.py'
SEARCH_SPEC = importlib.util.spec_from_file_location('search_exploration_coef', SEARCH_SCRIPT)
SEARCH = importlib.util.module_from_spec(SEARCH_SPEC)
SEARCH_SPEC.loader.exec_module(SEARCH)


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

	def test_coef_search_command_respects_video_config(self):
		command = SEARCH.command(
			'toy-bimodal',
			'q_bald',
			1e5,
			1,
			12_000,
			10_000,
			1_000,
			'coef-search',
			[],
		)
		self.assertFalse(any(arg.startswith('save_video=') for arg in command))

	def test_coef_search_summarizes_peak_region(self):
		rows = [
			{
				'plan/explore_coefficient': 80.,
				'plan/active_explore_bonus_task_ratio': 0.1,
				'plan/suggested_q_bald_coefficient': 180.,
			},
			{
				'plan/explore_coefficient': 100.,
				'plan/active_explore_bonus_task_ratio': 0.3,
				'plan/suggested_q_bald_coefficient': 220.,
			},
			{
				'eval/episode_reward': 12.,
				'eval/episode_reward_auc': 42.,
				'eval/metric_global_reached': 1.,
			},
		]
		result = SEARCH.summarize_history(rows, 'q_bald', 100., 0.2)
		self.assertAlmostEqual(result.peak_ratio, 0.2)
		self.assertAlmostEqual(result.median_suggested_coef, 200.)
		self.assertEqual(result.final_reward, 12.)
		self.assertEqual(result.global_success, 1.)

	def test_coef_search_prefers_success_inside_ratio_band(self):
		results = [
			SEARCH.Result(
				'q_bald', 1e4, 'a', 'a',
				reward_auc=100., global_success=0., peak_ratio=0.2, ratio_error=0.,
			),
			SEARCH.Result(
				'q_bald', 1e5, 'b', 'b',
				reward_auc=80., global_success=1., peak_ratio=0.25,
				ratio_error=SEARCH.math.log(1.25),
			),
		]
		self.assertEqual(SEARCH.select_best(results, 0.2).peak, 1e5)

	def test_fine_peaks_are_half_decade_neighbors(self):
		peaks = SEARCH.fine_peaks(1e5, [1e4, 1e5, 1e6])
		self.assertEqual(len(peaks), 2)
		self.assertAlmostEqual(peaks[0], 1e5 / SEARCH.math.sqrt(10.))
		self.assertAlmostEqual(peaks[1], 1e5 * SEARCH.math.sqrt(10.))


if __name__ == '__main__':
	unittest.main()
