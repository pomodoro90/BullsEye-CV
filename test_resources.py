import subprocess
import sys
import time
import unittest

import psutil

from log_resources import ProcTracker

CHILD = "import time\nt=time.time()\nwhile time.time()-t<6: sum(range(20000))\n"
PARENT = ("import subprocess, sys, time\n"
          f"c = subprocess.Popen([sys.executable, '-c', {CHILD!r}])\n"
          "time.sleep(7)\n")


class ProcTrackerTest(unittest.TestCase):
    def test_child_cpu_is_measured(self):
        p = subprocess.Popen([sys.executable, "-c", PARENT])
        try:
            time.sleep(1.5)
            ncpu = psutil.cpu_count() or 1
            tr = ProcTracker(psutil.Process(p.pid), ncpu)
            time.sleep(1.0)
            vals = []
            for _ in range(3):
                time.sleep(1.0)
                vals.append(tr.sample())
            cpu = max(v[0] for v in vals)

            self.assertGreater(cpu, 0.4 * 100 / ncpu, vals)
            self.assertGreater(vals[-1][1], 5.0)
            self.assertGreaterEqual(len(tr.procs), 2)
        finally:
            p.kill()
            for c in psutil.Process(p.pid).children(recursive=True) if psutil.pid_exists(p.pid) else []:
                c.kill()


if __name__ == "__main__":
    unittest.main()
