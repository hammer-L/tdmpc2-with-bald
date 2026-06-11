# TD-MPC2 BALD Toy Task 使用说明

本文档介绍两个用于快速验证 BALD 探索奖励的连续控制环境：

- `toy-bimodal`
- `toy-bimodal-dynamics`

它们都直接接入 TD-MPC2 的训练、评估、wandb 日志和视频链路，不需要 MuJoCo。

## 1. 环境定义

### 1.1 状态与动作

两个任务使用相同接口：

```text
observation = [position, velocity, normalized_time]
action      = [continuous_force], 范围 [-1, 1]
episode     = 50 steps
```

- `position`：一维轨道上的位置，范围约为 `[-1.5, 1.5]`。
- `velocity`：当前速度。
- `normalized_time`：当前步数除以 50。
- 环境本身不产生真实 termination，每局由 50 步 timeout 结束。

### 1.2 奖励结构

轨道上有两个奖励区域：

- 左侧蓝色宽区域：容易找到，但最高单步奖励只有约 `0.25`。
- 右侧绿色窄区域：难以随机找到，但最高单步奖励约为 `1.0`。

智能体只在左侧停留会得到稳定但较低的回报；找到并利用右侧区域才是全局更优行为。

### 1.3 两个任务的区别

`toy-bimodal` 使用全局一致的近似线性动力学，主要用于判断 Q-BALD 能否帮助发现远处高奖励区域。

`toy-bimodal-dynamics` 在 `position > 0.25` 后进入浅紫色区域，动作响应变为确定性的非线性函数：

- 最大向右动作 `action=1` 会失败。
- 约 `action=0.33` 的动作可以继续向右。
- 环境不加入随机动力学噪声，因此这里主要测试 epistemic uncertainty，而不是 aleatoric noise。

## 2. 视频画面

`env.render()` 返回固定尺寸的 `256×640×3` RGB 帧。画面元素如下：

- 深灰线：一维轨道。
- 深灰小车：智能体当前位置。
- 红色短线：当前速度方向和大致大小。
- 蓝色宽区域：局部低奖励区域。
- 绿色窄区域：全局高奖励目标。
- 浅紫色条纹：仅在 `toy-bimodal-dynamics` 中出现，表示非线性动力学区域。
- 顶部蓝色进度条：当前 episode 的 50 步进度。
- 到达全局目标后，小车和目标区域会变成更明显的绿色。

渲染画面使用 CPU PyTorch `torch.uint8` tensor 构造，最后转换成 NumPy RGB 帧，以兼容 TD-MPC2 原有的 DMControl 视频接口。

## 3. wandb 视频配置

视频只由 `tdmpc2/config.yaml` 中的配置控制：

```yaml
# logging
wandb_project: td2-toy
wandb_entity: your-wandb-entity
wandb_silent: false
enable_wandb: true

# misc
save_video: true
```

需要同时满足：

1. `enable_wandb=true`
2. `save_video=true`
3. `wandb_project` 和 `wandb_entity` 有效
4. 当前机器已经执行过 `wandb login`

训练每次触发周期评估时，只录制第一个 evaluation episode。视频沿用 TD-MPC2 的现有流程：

```text
env.render()
  -> VideoRecorder
  -> wandb.Video(format="mp4", fps=15)
  -> videos/eval_video
```

在 wandb run 页面进入 **Media**，查找 `videos/eval_video`。

关闭视频：

```bash
python tdmpc2/train.py task=toy-bimodal save_video=false
```

命令行参数会覆盖 `config.yaml`。实验矩阵脚本本身不会再强制修改 `save_video`。

## 4. 单组训练

以下命令都在仓库根目录执行。

### 4.1 无探索基线

```bash
python tdmpc2/train.py \
  task=toy-bimodal \
  model_size=5 \
  steps=30000 \
  explore_reward=none \
  exp_name=none
```

### 4.2 Q-BALD

```bash
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
```

### 4.3 Dynamics-BALD

```bash
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
```

### 4.4 随机 noise 对照

```bash
python tdmpc2/train.py \
  task=toy-bimodal \
  model_size=5 \
  steps=30000 \
  explore_reward=noise \
  explore_noise_std=1 \
  explore_schedule=triangular \
  explore_coef_peak=1 \
  exp_name=noise-triangular
```

## 5. 三角探索系数

默认三角调度为：

```text
seed collection 结束：coefficient = 0
调度进度 20%：       coefficient = explore_coef_peak
调度进度 100%：      coefficient = 0
之后：                coefficient = 0
```

相关配置：

```yaml
explore_schedule: triangular
explore_coef_peak: 1.0
explore_schedule_start: 0
explore_schedule_steps: 20_000
explore_peak_fraction: 0.2
```

探索奖励只用于 MPPI 候选 trajectory 的排序，不进入环境 reward、TD target、Q loss 或 reward-model loss。

## 6. 系数校准与完整实验矩阵

先打印 calibration 命令，不执行：

```bash
python scripts/run_exploration_experiments.py \
  --phase calibration \
  --q-peak 1 \
  --dynamics-peak 1 \
  --noise-peak 1 \
  --dry-run
```

移除 `--dry-run` 后会分别测试每种方法峰值的：

```text
0.5x, 1x, 2x
```

建议在峰值附近观察：

```text
train/explore_bonus_task_ratio
```

优先选择该比例约为 `0.2` 的 coefficient，再比较 `eval/episode_reward_auc` 和全局目标成功率。

执行完整 toy matrix：

```bash
python scripts/run_exploration_experiments.py \
  --phase toy \
  --q-peak 1 \
  --dynamics-peak 1 \
  --noise-peak 1
```

该命令顺序执行：

- `toy-bimodal`：5 种设置 × 3 seeds = 15 组
- `toy-bimodal-dynamics`：4 种设置 × 3 seeds = 12 组
- 总计 27 组

如果 `config.yaml` 中 `save_video=true`，这些运行会按各自的 `eval_freq` 上传评估视频。若只想先快速跑数值实验，可显式覆盖：

```yaml
save_video: false
```

矩阵脚本不覆盖该配置；单组 `train.py` 运行仍可直接使用 Hydra 参数 `save_video=false`。

## 7. 关键 wandb 指标

环境表现：

- `train/episode_reward`：训练 episode 总奖励。
- `eval/episode_reward`：关闭探索奖励后的评估回报。
- `eval/episode_reward_auc`：评估回报曲线的梯形积分。
- `train/metric_global_reached`：是否到达右侧全局目标。
- `train/metric_global_steps`：在全局目标区域停留步数。
- `train/metric_local_steps`：在左侧局部奖励区停留步数。
- `train/metric_max_position`：episode 到达的最右位置。
- `train/metric_state_coverage`：访问过的一维位置区间比例。

探索统计：

- `train/explore_coefficient`：当前探索系数。
- `train/explore_schedule_progress`：调度进度。
- `train/explore_reward_mean`：单步原始探索奖励均值。
- `train/explore_return_mean`：trajectory 原始探索回报均值。
- `train/explore_bonus_mean`：乘 coefficient 后的探索 bonus。
- `train/explore_bonus_task_ratio`：探索 bonus 与 task value 的相对尺度。
- `train/q_bald_mean` 或 `train/dynamics_bald_mean`：具体不确定性指标。

逐 planning 诊断默认配置：

```yaml
bald_diagnostics: true
plan_log_freq: 10
plan_alignment_target: 0.2
```

无论 `explore_reward` 使用 `none`、`q_bald`、`dynamics_bald` 还是 `noise`，每 10 次 planning 都会在最终 elite trajectories 上记录：

- `plan/elite_model_reward_return_mean`：规划窗口内 reward model 的折扣回报。
- `plan/elite_terminal_q_mean`：规划窗口末端的折扣 Q。
- `plan/elite_task_return_mean`：前两者之和，是 coefficient 的主要尺度基准。
- `plan/q_bald_return_mean`：Q-BALD 在规划窗口内的折扣累计值。
- `plan/dynamics_bald_return_mean`：dynamics-BALD 的折扣累计值。
- `plan/active_explore_bonus_task_ratio`：当前实际探索 bonus 与 elite task return 的比例。
- `plan/suggested_q_bald_coefficient`：使 Q-BALD bonus 约为 task return 20% 的建议系数。
- `plan/suggested_dynamics_bald_coefficient`：对应 dynamics-BALD 的建议系数。

建议系数由 `plan_alignment_target` 控制；BALD return 接近 0 时记录为 `NaN`。这些诊断在动作选定后计算，并恢复计算前的随机数状态，因此不会改变 MPPI 采样结果。

`train/*` 会给出 episode 级均值，`eval/*` 会给出周期评估均值。评估调用中探索 coefficient 恒为 0，但仍计算双 BALD，因此 `eval/episode_reward` 反映环境奖励策略，同时可以观察确定性评估策略的不确定性。

## 8. 从 checkpoint 生成本地 MP4

训练时视频上传 wandb。如果希望从 checkpoint 保存本地 MP4：

```bash
python tdmpc2/evaluate.py \
  task=toy-bimodal \
  model_size=5 \
  checkpoint=/path/to/final.pt \
  eval_episodes=3 \
  save_video=true \
  exp_name=toy-video
```

输出目录：

```text
logs/toy-bimodal/<seed>/toy-video/videos/
```

每个 evaluation episode 会生成一个 MP4。

## 9. 推荐判断标准

首轮建议使用 3 个随机种子。只有同时满足以下趋势时，再进入更昂贵的标准环境：

1. 三角调度前 10% 的环境回报相对无探索基线下降不超过约 10%。
2. BALD 在至少 2/3 seeds 上比 `none` 和 `noise` 更频繁到达全局目标。
3. coefficient 下降到 0 后，最后 20% 的环境回报没有明显回落。
4. `toy-bimodal-dynamics` 的所有对照都使用相同的 `dynamics_dropout=0.1`。

## 10. 常见问题

### wandb 中没有视频

检查：

```yaml
enable_wandb: true
save_video: true
wandb_project: <有效项目>
wandb_entity: <有效账号或团队>
```

并确认：

```bash
wandb login
```

视频只在 evaluation 时生成。如果尚未到达 `eval_freq`，wandb 中不会出现新视频。

### 27 组实验视频太多

将 `config.yaml` 暂时改为：

```yaml
save_video: false
```

筛选出最佳设置后，再单独用 `save_video=true` 复现实验。

### MP4 编码失败

确认项目环境中存在：

```bash
pip install imageio imageio-ffmpeg
```

如果系统仍找不到编码器，可使用：

```bash
conda install -c conda-forge ffmpeg
```

wandb 上传使用 `wandb.Video`；checkpoint 本地评估使用 `imageio.mimsave`。

### 视频正常但 wandb 标量没有明显提升

视频适合检查行为模式，但结论应以 3-seed 的 `episode_reward_auc`、`metric_global_reached` 和最终回报为主。

## 11. 测试

在 TD-MPC2 conda 环境执行：

```bash
python -m unittest discover -s tests -v
```

测试覆盖渲染尺寸与 dtype、状态变化、动力学区域标记、成功颜色、wandb VideoRecorder 和实验脚本的 `save_video` 行为。
