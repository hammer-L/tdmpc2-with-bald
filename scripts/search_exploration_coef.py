import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'tdmpc2' / 'train.py'
CONFIG = ROOT / 'tdmpc2' / 'config.yaml'

DEFAULT_PEAKS = {
	'q_bald': (1e4, 1e5, 1e6),
	'dynamics_bald': (1e3, 1e4, 1e5),
}
DEFAULT_TASKS = {
	'q_bald': 'toy-bimodal',
	'dynamics_bald': 'toy-bimodal-dynamics',
}


@dataclass
class Result:
	method: str
	peak: float
	run_id: str
	exp_name: str
	final_reward: float = float('nan')
	reward_auc: float = float('nan')
	global_success: float = float('nan')
	peak_ratio: float = float('nan')
	ratio_error: float = float('inf')
	median_suggested_coef: float = float('nan')
	run_url: str = ''


def config_value(key, default=None):
	"""Read a scalar from config.yaml without adding a YAML dependency."""
	prefix = f'{key}:'
	for line in CONFIG.read_text(encoding='utf-8').splitlines():
		stripped = line.strip()
		if stripped.startswith(prefix):
			value = stripped[len(prefix):].split('#', 1)[0].strip()
			return value or default
	return default


def peak_token(peak):
	"""Return a Hydra- and wandb-safe compact coefficient label."""
	return f'{peak:.6g}'.replace('+', '').replace('.', 'p')


def command(task, method, peak, seed, steps, schedule_steps, eval_freq, exp_name, overrides):
	args = [
		sys.executable,
		str(TRAIN),
		f'task={task}',
		'model_size=5',
		f'steps={steps}',
		'batch_size=256',
		'num_samples=128',
		'num_elites=16',
		'num_pi_trajs=8',
		'iterations=3',
		'horizon=5',
		f'eval_freq={eval_freq}',
		f'seed={seed}',
		f'explore_reward={method}',
		'explore_schedule=triangular',
		f'explore_coef_peak={peak:g}',
		f'explore_schedule_steps={schedule_steps}',
		'explore_peak_fraction=0.2',
		'dynamics_dropout=0.1',
		'bald_diagnostics=true',
		'plan_log_freq=10',
		'plan_alignment_target=0.2',
		f'exp_name={exp_name}',
	]
	args.extend(overrides)
	return args


def finite(value):
	try:
		value = float(value)
	except (TypeError, ValueError):
		return float('nan')
	return value if math.isfinite(value) else float('nan')


def median(values):
	values = [finite(value) for value in values]
	values = [value for value in values if math.isfinite(value)]
	return statistics.median(values) if values else float('nan')


def score_value(value, default=float('-inf')):
	value = finite(value)
	return value if math.isfinite(value) else default


def summarize_history(rows, method, peak, target, run_id='', exp_name='', run_url=''):
	"""Summarize wandb history into coefficient-search signals."""
	coef_key = 'plan/explore_coefficient'
	ratio_key = 'plan/active_explore_bonus_task_ratio'
	suggested_key = f'plan/suggested_{method}_coefficient'
	reward_key = 'eval/episode_reward'
	auc_key = 'eval/episode_reward_auc'
	success_key = 'eval/metric_global_reached'

	near_peak_ratios, suggested = [], []
	rewards, aucs, successes = [], [], []
	for row in rows:
		coef = finite(row.get(coef_key))
		ratio = finite(row.get(ratio_key))
		if (
			math.isfinite(coef)
			and math.isfinite(ratio)
			and peak > 0
			and coef >= 0.8 * peak
		):
			near_peak_ratios.append(ratio)
		value = finite(row.get(suggested_key))
		if math.isfinite(value):
			suggested.append(value)
		value = finite(row.get(reward_key))
		if math.isfinite(value):
			rewards.append(value)
		value = finite(row.get(auc_key))
		if math.isfinite(value):
			aucs.append(value)
		value = finite(row.get(success_key))
		if math.isfinite(value):
			successes.append(value)

	peak_ratio = median(near_peak_ratios)
	ratio_error = (
		abs(math.log(peak_ratio / target))
		if peak_ratio > 0 and target > 0
		else float('inf')
	)
	return Result(
		method=method,
		peak=peak,
		run_id=run_id,
		exp_name=exp_name,
		final_reward=rewards[-1] if rewards else float('nan'),
		reward_auc=aucs[-1] if aucs else float('nan'),
		global_success=max(successes) if successes else float('nan'),
		peak_ratio=peak_ratio,
		ratio_error=ratio_error,
		median_suggested_coef=median(suggested),
		run_url=run_url,
	)


def select_best(results, target):
	"""Prefer target-scale candidates, then success and reward AUC."""
	valid = [result for result in results if math.isfinite(result.peak_ratio)]
	if not valid:
		raise RuntimeError('No run produced a finite peak bonus/task ratio.')
	in_band = [
		result
		for result in valid
		if 0.5 * target <= result.peak_ratio <= 2.0 * target
	]
	if in_band:
		return max(
			in_band,
			key=lambda result: (
				score_value(result.global_success),
				score_value(result.reward_auc),
				-result.ratio_error,
			),
		)
	return min(
		valid,
		key=lambda result: (
			result.ratio_error,
			-score_value(result.global_success),
			-score_value(result.reward_auc),
		),
	)


def fine_peaks(best_peak, existing):
	"""Return two half-decade neighbors around a coarse winner."""
	factor = math.sqrt(10.)
	candidates = (best_peak / factor, best_peak * factor)
	return [
		peak
		for peak in candidates
		if not any(math.isclose(peak, old, rel_tol=1e-9) for old in existing)
	]


def fetch_wandb_result(entity, project, run_id, method, peak, target, exp_name):
	import wandb

	api = wandb.Api(timeout=60)
	last_error = None
	for _ in range(12):
		try:
			run = api.run(f'{entity}/{project}/{run_id}')
			rows = list(run.scan_history(page_size=1000))
			return summarize_history(
				rows,
				method,
				peak,
				target,
				run_id=run_id,
				exp_name=exp_name,
				run_url=run.url,
			)
		except Exception as error:
			last_error = error
			time.sleep(5)
	raise RuntimeError(f'Failed to read wandb run {run_id}: {last_error}')


def run_candidate(args, method, peak):
	task = args.q_task if method == 'q_bald' else args.dynamics_task
	exp_name = f'{args.prefix}-{method}-p{peak_token(peak)}'
	run_id = uuid.uuid4().hex[:8]
	cmd = command(
		task,
		method,
		peak,
		args.seed,
		args.steps,
		args.schedule_steps,
		args.eval_freq,
		exp_name,
		args.override,
	)
	print(f'\n[{method}] peak={peak:g} run_id={run_id}', flush=True)
	print(' '.join(f'"{part}"' if ' ' in part else part for part in cmd), flush=True)
	if args.dry_run:
		return None
	env = os.environ.copy()
	env['WANDB_RUN_ID'] = run_id
	env['WANDB_RESUME'] = 'never'
	subprocess.run(cmd, cwd=ROOT, env=env, check=True)
	result = fetch_wandb_result(
		args.entity,
		args.project,
		run_id,
		method,
		peak,
		args.target_ratio,
		exp_name,
	)
	print_result(result)
	return result


def print_result(result):
	print(
		f'  ratio={result.peak_ratio:.4g} '
		f'success={result.global_success:.4g} '
		f'final_reward={result.final_reward:.4g} '
		f'auc={result.reward_auc:.4g} '
		f'suggested_median={result.median_suggested_coef:.4g}',
		flush=True,
	)


def print_table(results, target):
	print('\nCoefficient ranking:')
	print('method          peak       peak_ratio  success   final_R      AUC')
	for result in sorted(results, key=lambda item: (item.method, item.peak)):
		marker = '*' if 0.5 * target <= result.peak_ratio <= 2 * target else ' '
		print(
			f'{marker}{result.method:<15} '
			f'{result.peak:<10.4g} '
			f'{result.peak_ratio:<11.4g} '
			f'{result.global_success:<9.4g} '
			f'{result.final_reward:<12.4g} '
			f'{result.reward_auc:<12.4g}'
		)
	print('* ratio lies within [0.5x, 2x] of the target.')


def save_results(results, output):
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
		writer.writeheader()
		writer.writerows(asdict(result) for result in results)


def parse_args():
	default_prefix = f'coef-search-{datetime.now():%Y%m%d-%H%M%S}'
	parser = argparse.ArgumentParser(
		description='Coarse-to-fine search for TD-MPC2 BALD coefficients.'
	)
	parser.add_argument(
		'--method',
		choices=['q_bald', 'dynamics_bald', 'both'],
		default='both',
	)
	parser.add_argument('--q-peaks', type=float, nargs='+', default=DEFAULT_PEAKS['q_bald'])
	parser.add_argument(
		'--dynamics-peaks',
		type=float,
		nargs='+',
		default=DEFAULT_PEAKS['dynamics_bald'],
	)
	parser.add_argument('--q-task', default=DEFAULT_TASKS['q_bald'])
	parser.add_argument('--dynamics-task', default=DEFAULT_TASKS['dynamics_bald'])
	parser.add_argument('--steps', type=int, default=12_000)
	parser.add_argument('--schedule-steps', type=int, default=10_000)
	parser.add_argument('--eval-freq', type=int, default=1_000)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--target-ratio', type=float, default=0.2)
	parser.add_argument('--coarse-only', action='store_true')
	parser.add_argument('--prefix', default=default_prefix)
	parser.add_argument('--project', default=config_value('wandb_project', 'td2-toy'))
	parser.add_argument('--entity', default=config_value('wandb_entity'))
	parser.add_argument(
		'--override',
		action='append',
		default=[],
		help='Additional Hydra override; repeat as needed, e.g. --override save_video=false.',
	)
	parser.add_argument('--dry-run', action='store_true')
	return parser.parse_args()


def main():
	args = parse_args()
	if not args.dry_run and (not args.entity or not args.project):
		raise ValueError('wandb entity and project are required.')
	if args.target_ratio <= 0:
		raise ValueError('--target-ratio must be positive.')

	methods = ['q_bald', 'dynamics_bald'] if args.method == 'both' else [args.method]
	grids = {
		'q_bald': list(args.q_peaks),
		'dynamics_bald': list(args.dynamics_peaks),
	}
	results = []
	for method in methods:
		method_results = []
		for peak in grids[method]:
			result = run_candidate(args, method, peak)
			if result is not None:
				method_results.append(result)
				results.append(result)
		if args.dry_run:
			continue
		coarse_best = select_best(method_results, args.target_ratio)
		print(f'\nCoarse winner for {method}: {coarse_best.peak:g}')
		if not args.coarse_only:
			for peak in fine_peaks(coarse_best.peak, grids[method]):
				result = run_candidate(args, method, peak)
				method_results.append(result)
				results.append(result)
		best = select_best(method_results, args.target_ratio)
		print(
			f'\nRecommended {method} coefficient: {best.peak:g} '
			f'(peak ratio={best.peak_ratio:.4g}, run={best.run_url})'
		)

	if args.dry_run:
		return
	print_table(results, args.target_ratio)
	output = ROOT / 'results' / f'{args.prefix}.csv'
	save_results(results, output)
	print(f'\nSaved results to {output}')


if __name__ == '__main__':
	main()
