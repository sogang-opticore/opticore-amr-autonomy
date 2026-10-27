import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_pilot


class PilotTest(unittest.TestCase):
    def test_exact_train_pairs_and_alternated_order(self):
        tasks = run_pilot.schedule()
        self.assertEqual(len(set(tasks)), 16)
        for idx in range(0, 8, 2):
            mission = tasks[idx][0]
            self.assertTrue(mission.startswith("train-"))
            self.assertEqual(tasks[idx:idx+2], [(mission, "b0", 0), (mission, "b1", 0)])
            self.assertEqual(tasks[8+idx:10+idx], [(mission, "b1", 1), (mission, "b0", 1)])

    def test_interrupt_keeps_partial_batch_summary_and_never_reuses_output(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(tmp) / "batch"
            with patch("run_pilot.acquire", side_effect=KeyboardInterrupt):
                self.assertEqual(run_pilot.run_pilot(out, 83), 130)
            report = json.loads((out / "summary.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["error"], "pilot interrupted")
            self.assertEqual(len(report["input_directories"]), 1)
            self.assertEqual(len(list(out.glob("*.log"))), 1)
            with self.assertRaises(FileExistsError):
                run_pilot.run_pilot(out, 83)

    def test_abort_after_failed_second_run(self):
        def fake_runner(cmd, log):
            trial = Path(cmd[-1])
            trial.mkdir()
            (trial / "evidence").write_text("retained")
            return int("b1" in cmd)

        def fake_summary(directories, splits, seeds):
            entries = [{"disposition": "runner_error" if "-b1-" in d.name else "collision",
                        "detail": "fixture"} for d in directories]
            return {"input_directories": entries, "planners": {
                "b0": {"expected": 8, "scored": 1, "counts": {"collision":1, "missing":7}},
                "b1": {"expected": 8, "scored": 0, "counts": {"runner_error":1, "missing":7}}}}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(tmp) / "batch"
            with patch("run_pilot.acquire", side_effect=fake_runner) as run, \
                    patch("run_pilot.summarize", side_effect=fake_summary):
                self.assertEqual(run_pilot.run_pilot(out, 84), 1)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(len(list(out.glob("*/evidence"))), 2)
            report = json.loads((out / "summary.json").read_text())
            self.assertIn("runner exit 1", report["error"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["planners"]["b0"]["counts"]["collision"], 1)
