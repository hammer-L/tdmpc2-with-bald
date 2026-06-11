import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tdmpc2'))

try:
	from common.logger import Logger, VideoRecorder
except ImportError:
	Logger = None
	VideoRecorder = None


class FakeEnv:
	def __init__(self):
		self.render_calls = 0

	def render(self):
		self.render_calls += 1
		return np.zeros((16, 24, 3), dtype=np.uint8)


class FakeWandb:
	def __init__(self):
		self.video_calls = []
		self.log_calls = []

	def Video(self, frames, fps, format):
		video = {'frames': frames, 'fps': fps, 'format': format}
		self.video_calls.append(video)
		return video

	def log(self, payload, step):
		self.log_calls.append((payload, step))
		return payload


@unittest.skipIf(VideoRecorder is None, 'logger dependencies are not installed')
class VideoRecorderTest(unittest.TestCase):

	def test_records_episode_and_uploads_mp4(self):
		with tempfile.TemporaryDirectory() as tmp:
			wandb = FakeWandb()
			recorder = VideoRecorder(SimpleNamespace(work_dir=Path(tmp)), wandb)
			env = FakeEnv()
			recorder.init(env)
			recorder.record(env)
			recorder.save(step=123)

		self.assertEqual(env.render_calls, 2)
		self.assertEqual(len(wandb.video_calls), 1)
		self.assertEqual(wandb.video_calls[0]['frames'].shape, (2, 3, 16, 24))
		self.assertEqual(wandb.video_calls[0]['fps'], 15)
		self.assertEqual(wandb.video_calls[0]['format'], 'mp4')
		self.assertEqual(wandb.log_calls[0][1], 123)
		self.assertIn('videos/eval_video', wandb.log_calls[0][0])

	def test_disabled_recording_does_not_render(self):
		with tempfile.TemporaryDirectory() as tmp:
			recorder = VideoRecorder(SimpleNamespace(work_dir=Path(tmp)), FakeWandb())
			env = FakeEnv()
			recorder.init(env, enabled=False)
			recorder.record(env)
		self.assertEqual(env.render_calls, 0)


@unittest.skipIf(Logger is None, 'logger dependencies are not installed')
class PlanLoggerTest(unittest.TestCase):

	def test_plan_log_uses_global_step_without_console_or_csv(self):
		logger = object.__new__(Logger)
		logger._wandb = FakeWandb()
		logger.log_plan({'elite_task_return_mean': 3.5}, step=20)
		payload, step = logger._wandb.log_calls[0]
		self.assertEqual(step, 20)
		self.assertEqual(payload, {'plan/elite_task_return_mean': 3.5})


if __name__ == '__main__':
	unittest.main()
