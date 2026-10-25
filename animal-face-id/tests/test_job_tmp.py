"""Regression checks for the AF_UNIX failure in the GPU job launchers."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class JobTemporaryDirectoryTests(unittest.TestCase):
    def test_original_long_tmpdir_breaks_descriptor_sharing(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_dir = Path(tmp) / ("long_suite_name_" * 12)
            long_dir.mkdir()
            result = subprocess.run(
                [sys.executable, "-c",
                 "import os; from multiprocessing.reduction import DupFd; "
                 "DupFd(os.open(os.devnull, os.O_RDONLY)).detach()"],
                env={**os.environ, "TMPDIR": str(long_dir), "TEMP": str(long_dir), "TMP": str(long_dir)},
                capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("AF_UNIX path too long", result.stderr)

    def test_launcher_uses_short_tmpdir_passes_ipc_check_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_dir = Path(tmp) / ("long_suite_name_" * 12)
            long_dir.mkdir()
            for exit_code in (0, 7):
                result = subprocess.run(
                    ["bash", "-euc",
                     'python() { "$MACAQUE_TEST_PYTHON" "$@"; }; '
                     'source "$1"; printf "RESULT_TMPDIR=%s\\n" "$TMPDIR"; exit "$2"',
                     "tmp-check", str(ROOT / "scripts/setup_job_tmp.sh"), str(exit_code)],
                    env={**os.environ, "TMPDIR": str(long_dir), "SLURM_TMPDIR": str(long_dir),
                         "SLURM_JOB_ID": "12345", "MACAQUE_TEST_PYTHON": sys.executable},
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertIn("Multiprocessing IPC check passed", result.stdout)
                directory = next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                                 if line.startswith("RESULT_TMPDIR="))
                self.assertTrue(directory.startswith("/tmp/mq.12345."))
                self.assertLess(len(directory), 50)
                self.assertFalse(Path(directory).exists(), "Job temporary directory was not cleaned up")


if __name__ == "__main__":
    unittest.main()
