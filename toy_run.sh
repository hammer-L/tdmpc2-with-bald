export CUDA_VISIBLE_DEVICES=1

# 1. baseline
python tdmpc2/train.py \
  task=toy-bimodal \
  model_size=5 \
  steps=30000 \
  explore_reward=none \
  exp_name=none

# 2. Q-BALD
python tdmpc2/train.py \
  task=toy-bimodal \
  model_size=5 \
  steps=30000 \
  num_samples=128 \
  num_elites=16 \
  num_pi_trajs=8 \
  iterations=3 \
  horizon=5 \
  eval_freq=2000 \
  explore_reward=q_bald \
  explore_schedule=triangular \
  explore_coef_peak=1 \
  explore_schedule_steps=20000 \
  explore_peak_fraction=0.2 \
  exp_name=q-bald-triangular

# 3. Dynamics-BALD
python tdmpc2/train.py \
  task=toy-bimodal-dynamics \
  model_size=5 \
  steps=30000 \
  explore_reward=dynamics_bald \
  dynamics_dropout=0.1 \
  dynamics_bald_samples=5 \
  explore_schedule=triangular \
  explore_coef_peak=1 \
  explore_schedule_steps=20000 \
  explore_peak_fraction=0.2 \
  exp_name=dynamics-bald-triangular

# 4. random noise
python tdmpc2/train.py \
  task=toy-bimodal \
  model_size=5 \
  steps=30000 \
  explore_reward=noise \
  explore_noise_std=1 \
  explore_schedule=triangular \
  explore_coef_peak=1 \
  exp_name=noise-triangular