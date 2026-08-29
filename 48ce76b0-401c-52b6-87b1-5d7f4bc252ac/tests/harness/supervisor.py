"""Parent side of the S03 verifier's isolated re-execution.

The verifier runs the whole grid itself: every comparator draw and every
submitted arm, one child interpreter each, records streaming back on pipes the
parent created. The parent writes the record file, owns the checkpoint
directory, measures the final validation loss in a process that holds no
submitted byte, and counts the egress events off the stream rather than off a
file in the agent's workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CHILD = os.path.join(HERE, "train_child.py")
MEASURE_CHILD = os.path.join(HERE, "measure_child.py")
ORDER_CHILD = os.path.join(HERE, "order_child.py")


class ChildFailure(Exception):
    def __init__(self, code, detail):
        super().__init__(code)
        self.code = code
        self.detail = detail


class Supervisor:
    def __init__(self, bundle, profile, order_module):
        self.bundle = bundle
        self.profile = profile
        self.order_module = order_module
        self.root = tempfile.mkdtemp(prefix="bia-s03-verify-")
        self.ckpt = os.path.join(self.root, "ckpt")
        self.cache = os.path.join(self.root, "cache")
        self.scratch = os.path.join(self.root, "scratch")
        self.home = os.path.join(self.root, "home")
        self.cwd = os.path.join(self.root, "cwd")
        for d in (self.ckpt, self.cache, self.scratch, self.home, self.cwd):
            os.makedirs(d, exist_ok=True)
        self.record_path = os.path.join(self.root, "verifier_record.jsonl")
        self.child_log = os.path.join(self.root, "child_stdio.log")
        self.records = []
        self.egress_events = []
        self._seq = 0
        self._measure = None
        self._chan = None

    def child_env(self, extra=None):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = self.scratch
        env["HOME"] = self.home
        env["XDG_CACHE_HOME"] = os.path.join(self.home, ".cache")
        env["BIA_PROFILE"] = self.profile
        env.pop("BIA_TELEMETRY", None)
        env.pop("BIA_EGRESS_AUDIT", None)
        env.pop("BIA_SCORE_PATH", None)
        env.pop("BIA_OUTCOMES_PATH", None)
        if extra:
            env.update(extra)
        return env

    def append(self, body):
        """The parent assigns the sequence number. The child never sees it."""
        record = dict(body)
        record["seq"] = self._seq
        self._seq += 1
        self.records.append(record)
        return record

    def _stream(self, script, request, env=None):
        read_fd, write_fd = os.pipe()
        try:
            log = open(self.child_log, "a")
            proc = subprocess.Popen(
                [sys.executable, script, "--channel-fd", str(write_fd)],
                stdin=subprocess.PIPE, stdout=log, stderr=log,
                pass_fds=(write_fd,), cwd=self.cwd, env=env or self.child_env())
            os.close(write_fd)
            write_fd = None
            proc.stdin.write((json.dumps(request) + "\n").encode())
            proc.stdin.close()
            fault = None
            with os.fdopen(read_fd, "r") as chan:
                read_fd = None
                for line in chan:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("kind") == "fault":
                        fault = obj
                    elif obj.get("event") == "egress_attempt":
                        self.egress_events.append(obj)
                    else:
                        self.append(obj)
            rc = proc.wait()
        finally:
            for fd in (read_fd, write_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        return rc, fault

    def run_arm(self, arm, seed, draw):
        name = "%s_seed%d_draw%d.pt" % (arm, seed, draw)
        ckpt = os.path.join(self.ckpt, name)
        request = {
            "bundle": self.bundle, "profile": self.profile, "arm": arm,
            "seed": int(seed), "draw": int(draw),
            "order_module": self.order_module, "ckpt_path": ckpt,
            "cache_dir": self.cache,
        }
        rc, fault = self._stream(TRAIN_CHILD, request)
        if fault is not None:
            raise ChildFailure("arm-faulted", {"arm": arm, "seed": seed,
                                               "draw": draw, "fault": fault})
        if rc != 0:
            raise ChildFailure("arm-exit-nonzero", {"arm": arm, "seed": seed,
                                                    "draw": draw, "returncode": rc})
        return ckpt

    def start_measure(self):
        read_fd, write_fd = os.pipe()
        log = open(self.child_log, "a")
        proc = subprocess.Popen(
            [sys.executable, MEASURE_CHILD, "--channel-fd", str(write_fd)],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            pass_fds=(write_fd,), cwd=self.cwd, env=self.child_env())
        os.close(write_fd)
        self._measure = proc
        self._chan = os.fdopen(read_fd, "r")
        proc.stdin.write((json.dumps(
            {"bundle": self.bundle, "profile": self.profile,
             "cache_dir": self.cache}) + "\n").encode())
        proc.stdin.flush()
        hello = self._chan.readline()
        if not hello:
            raise ChildFailure("measure-child-did-not-start", {})
        doc = json.loads(hello)
        if doc.get("kind") != "ready":
            raise ChildFailure("measure-child-fault", doc)
        return doc

    def measure(self, ckpt):
        self._measure.stdin.write((json.dumps({"ckpt": ckpt}) + "\n").encode())
        self._measure.stdin.flush()
        line = self._chan.readline()
        if not line:
            raise ChildFailure("measure-child-closed-channel", {"ckpt": ckpt})
        doc = json.loads(line)
        if doc.get("kind") != "measured":
            raise ChildFailure("measure-child-fault", doc)
        return doc

    def stop_measure(self):
        if self._measure is None:
            return
        try:
            self._measure.stdin.close()
        except Exception:
            pass
        try:
            self._measure.wait(timeout=120)
        except Exception:
            self._measure.kill()
        try:
            self._chan.close()
        except Exception:
            pass
        self._measure = None

    def derive_order(self, identity, cwd):
        env = self.child_env({"BIA_GRADER_IDENTITY": identity,
                              "HOSTNAME": "bia-grader-" + identity})
        proc = subprocess.run(
            [sys.executable, ORDER_CHILD, "--bundle", self.bundle,
             "--profile", self.profile, "--order-module", self.order_module,
             "--cache-dir", self.cache],
            capture_output=True, text=True, env=env, cwd=cwd)
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1]), None
        except (json.JSONDecodeError, IndexError):
            return None, (proc.stdout + proc.stderr).strip()[-400:]

    def flush(self):
        with open(self.record_path, "w") as fh:
            for record in self.records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return self.record_path
