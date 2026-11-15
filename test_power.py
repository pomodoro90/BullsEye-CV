import tempfile
import unittest

import server


class PowerTest(unittest.TestCase):
    def test_argv(self):
        self.assertEqual(server.power_argv("shutdown", windows=True), ["shutdown", "/s", "/t", "15"])
        self.assertEqual(server.power_argv("restart", windows=True), ["shutdown", "/r", "/t", "15"])
        self.assertEqual(server.power_argv("cancel", windows=True), ["shutdown", "/a"])
        self.assertEqual(server.power_argv("shutdown", windows=False)[:2], ["shutdown", "-h"])

    def test_endpoint(self):
        from starlette.testclient import TestClient
        calls = []
        orig = server.run_power
        server.run_power = calls.append
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                S, app = server.build(server.parse_args(["--sim", "--data-dir", tmp]))
                with TestClient(app) as c:
                    self.assertEqual(c.post("/api/power", json={"action": "shutdown"}).status_code, 400)
                    S.power_enabled = True
                    self.assertEqual(c.post("/api/power", json={"action": "bogus"}).status_code, 400)
                    self.assertEqual(c.post("/api/power", json={"action": "shutdown"}).status_code, 200)
                    self.assertEqual(c.post("/api/power", json={"action": "cancel"}).status_code, 200)
        finally:
            server.run_power = orig
        self.assertEqual(calls, ["shutdown", "cancel"])


class QuitTest(unittest.TestCase):
    def test_quit_exits_with_code_42(self):
        import json
        import subprocess
        import sys
        import time
        import urllib.request
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            port = 8765
            p = subprocess.Popen([sys.executable, "server.py", "--sim", "--port", str(port), "--data-dir", tmp],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=1)
                        break
                    except Exception:
                        time.sleep(0.25)
                req = urllib.request.Request(f"http://localhost:{port}/api/power", data=json.dumps({"action": "quit"}).encode(),
                                             headers={"Content-Type": "application/json"})
                self.assertEqual(json.loads(urllib.request.urlopen(req).read())["ok"], True)
                self.assertEqual(p.wait(timeout=15), server.QUIT_EXIT_CODE)
            finally:
                if p.poll() is None:
                    p.kill()


if __name__ == "__main__":
    unittest.main()
