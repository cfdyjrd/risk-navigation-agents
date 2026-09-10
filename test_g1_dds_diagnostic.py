"""Exercise the actual subprocess timeout without loading SDK or opening DDS."""
from contextlib import redirect_stderr
import io
import sys
import time
import unittest

from g1_dds_diagnostic import _run_supervised


class DiagnosticSupervisorTests(unittest.TestCase):
    def test_stuck_worker_is_terminated_with_timeout_status(self):
        stderr = io.StringIO()
        start = time.monotonic()
        with redirect_stderr(stderr):
            result = _run_supervised([sys.executable, "-c", "import time; time.sleep(60)"], .2)
        self.assertEqual(result, 124)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIn("诊断总超时", stderr.getvalue())

    def test_worker_exit_status_is_preserved(self):
        result = _run_supervised([sys.executable, "-c", "raise SystemExit(7)"], 5)
        self.assertEqual(result, 7)


if __name__ == "__main__":
    unittest.main()
