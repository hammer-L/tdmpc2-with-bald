import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'tdmpc2' / 'train.py'


def command(task, method, schedule, peak, seed, steps, schedule_steps, exp_name):
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
		'eval_freq=2000',
		f'seed={seed}',
		f'explore_reward={method}',
		f'explore_schedule={schedule}',
		f'explore_coef_peak={peak}',
		f'explore_schedule_steps={schedule_steps}',
		'explore_peak_fraction=0.2',
		'dynamics_dropout=0.1',
		f'exp_name={exp_name}',
	]
	if task == 'mountaincar-continuous':
		args.append('episodic=true')
	return args


def main():
	parser = argparse.ArgumentParser(description='Run TD-MPC2 BALD toy experiments.')
	parser.add_argument('--phase', choices=['calibration', 'toy', 'confirm'], default='toy')
	parser.add_argument('--q-peak', type=float, default=1.)
	parser.add_argument('--dynamics-peak', type=float, default=1.)
	parser.add_argument('--noise-peak', type=float, default=1.)
	parser.add_argument('--method', choices=['q_bald', 'dynamics_bald'], default='q_bald')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()

	peaks = {
		'none': 0.,
		'q_bald': args.q_peak,
		'dynamics_bald': args.dynamics_peak,
		'noise': args.noise_peak,
	}
	runs = []
	if args.phase == 'calibration':
		for method, task in [
			('q_bald', 'toy-bimodal'),
			('dynamics_bald', 'toy-bimodal-dynamics'),
			('noise', 'toy-bimodal'),
		]:
			for multiplier in (0.5, 1., 2.):
				peak = peaks[method] * multiplier
				name = f'calibrate-{method}-{multiplier:g}x'
				runs.append((task, method, 'triangular', peak, 1, 10_000, 8_000, name))
	elif args.phase == 'toy':
		matrix = {
			'toy-bimodal': [
				('none', 'triangular'),
				('q_bald', 'triangular'),
				('q_bald', 'constant'),
				('q_bald', 'linear_decay'),
				('noise', 'triangular'),
			],
			'toy-bimodal-dynamics': [
				('none', 'triangular'),
				('q_bald', 'triangular'),
				('dynamics_bald', 'triangular'),
				('noise', 'triangular'),
			],
		}
		for task, variants in matrix.items():
			for method, schedule in variants:
				for seed in (1, 2, 3):
					name = f'{method}-{schedule}'
					runs.append((task, method, schedule, peaks[method], seed, 30_000, 20_000, name))
	else:
		for task, steps in [
			('mountaincar-continuous', 100_000),
			('cartpole-swingup-sparse', 100_000),
		]:
			for method in ('none', args.method, 'noise'):
				for seed in (1, 2, 3):
					peak = peaks[method]
					name = f'confirm-{method}-triangular'
					runs.append((task, method, 'triangular', peak, seed, steps, 60_000, name))

	for run in runs:
		cmd = command(*run)
		print(' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd), flush=True)
		if not args.dry_run:
			import subprocess
			subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
	main()
