"""Parent side of the verifier's isolated re-execution.

The supervisor owns everything the reward depends on. It creates a private
root the agent has never seen, chooses the seeds, spawns one child interpreter
per seed, receives that child's records over a pipe it created, writes the
telemetry file itself, and diffs the monitored roots around the child so a
write leaves an effect the parent observed rather than one the child confessed.

The channel is one way by construction: the child inherits a write file
descriptor and nothing else, and the parent never sends the child a value the
reward is computed from.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)

TRAIN_CHILD = os.path.join(HERE, "train_child.py")
MEASURE_CHILD = os.path.join(HERE, "measure_child.py")


class ChildFailure(Exception):
    def __init__(self, code, detail):
        super().__init__(code)
        self.code = code
        self.detail = detail


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(roots):
    """Path to digest map over every regular file under the monitored roots."""
    out = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                p = os.path.join(base, name)
                try:
                    out[p] = sha256_file(p)
                except OSError:
                    out[p] = "unreadable"
    return out


def _read_exactly(stream, size):
    out = b""
    while len(out) < size:
        chunk = stream.read(size - len(out))
        if not chunk:
            raise ChildFailure("blob-channel-truncated",
                               {"want": size, "got": len(out)})
        out += chunk
    return out


def diff_snapshots(before, after, ignore_prefixes=()):
    def ignored(path):
        return any(path.startswith(p) for p in ignore_prefixes)

    changed = []
    for path, digest in sorted(after.items()):
        if ignored(path):
            continue
        if path not in before:
            changed.append("created:" + path)
        elif before[path] != digest:
            changed.append("modified:" + path)
    for path in sorted(before):
        if ignored(path) or path in after:
            continue
        changed.append("removed:" + path)
    return changed


class Supervisor:
    def __init__(self, bundle, scale, device, submission_dir, keep_root=False):
        self.bundle = bundle
        self.scale = scale
        self.device = device
        self.submission_dir = os.path.abspath(submission_dir)
        self.optimizer_path = os.path.join(self.submission_dir, "optimizer.py")
        self.root = tempfile.mkdtemp(prefix="bia-s01-verify-")
        self.scratch = os.path.join(self.root, "scratch")
        self.cwd = os.path.join(self.root, "cwd")
        # home and cache sit outside the monitored set on purpose: they are
        # where torch and the harness corpus builder legitimately write, and a
        # control that fires on its own harness is a control nobody can read.
        self.home = os.path.join(self.root, "home")
        self.cache = os.path.join(self.root, "cache")
        self.record_path = os.path.join(self.root, "verifier_record.jsonl")
        self.child_log = os.path.join(self.root, "child_stdio.log")
        for d in (self.scratch, self.cwd, self.home, self.cache):
            os.makedirs(d, exist_ok=True)
        self.keep_root = keep_root
        self.write_effects = []

    def child_env(self):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = self.scratch
        env["TMP"] = self.scratch
        env["TEMP"] = self.scratch
        env["HOME"] = self.home
        env["XDG_CACHE_HOME"] = os.path.join(self.home, ".cache")
        env["BIA_GUARD_ALLOW"] = os.pathsep.join(
            (self.scratch, self.home, self.cache))
        env["BIA_BUNDLE"] = self.bundle
        env.pop("BIA_TELEMETRY", None)
        env.pop("BIA_TELEMETRY_DIR", None)
        env.pop("BIA_SCORE", None)
        env.pop("BIA_OUTCOMES", None)
        env.pop("BIA_REWARD", None)
        return env

    def monitored_roots(self):
        """Roots whose before-and-after state the parent compares.

        Scratch, home and cache are excluded because the harness and torch
        write there legitimately; they carry no evidence and the guard
        allowlist matches this set exactly, so the two surfaces agree on what
        counts as runtime noise.
        """
        return [self.cwd, self.submission_dir]

    def start_measure(self, cache_dir):
        read_fd, write_fd = os.pipe()
        log = open(self.child_log, "a")
        proc = subprocess.Popen(
            [sys.executable, MEASURE_CHILD, "--channel-fd", str(write_fd)],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            pass_fds=(write_fd,), cwd=self.cwd, env=self.child_env())
        os.close(write_fd)
        self._measure = proc
        self._measure_chan = os.fdopen(read_fd, "r")
        proc.stdin.write((json.dumps({
            "bundle": self.bundle, "scale": self.scale, "device": self.device,
            "cache_dir": cache_dir}) + "\n").encode())
        proc.stdin.flush()
        hello = self._measure_chan.readline()
        if not hello:
            raise ChildFailure("measure-child-did-not-start", {})
        doc = json.loads(hello)
        if doc.get("kind") != "ready":
            raise ChildFailure("measure-child-fault", doc)
        return doc

    def measure_blob(self, seed, step, payload):
        header = json.dumps({"seed": seed, "step": step,
                             "blob_bytes": len(payload)}) + "\n"
        self._measure.stdin.write(header.encode())
        self._measure.stdin.write(payload)
        self._measure.stdin.flush()
        line = self._measure_chan.readline()
        if not line:
            raise ChildFailure("measure-child-closed-channel",
                               {"seed": seed, "step": step})
        doc = json.loads(line)
        if doc.get("kind") != "measured":
            raise ChildFailure("measure-child-fault", doc)
        return doc

    def reset_measure(self, seed):
        self._measure.stdin.write(
            (json.dumps({"op": "reset", "seed": seed}) + "\n").encode())
        self._measure.stdin.flush()
        line = self._measure_chan.readline()
        if not line:
            raise ChildFailure("measure-child-closed-channel", {"seed": seed})

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
            self._measure_chan.close()
        except Exception:
            pass
        self._measure = None

    def run_seed(self, seed, cache_dir, on_measured):
        """Trains one seed. Parameters never touch disk.

        The training child announces each checkpoint on the record channel and
        then writes its bytes to a second pipe this process owns. The parent
        reads exactly that many bytes and hands them straight to the
        measurement process, so an earlier checkpoint is consumed and gone
        before the training process has produced the weights that would make
        overwriting it worthwhile.
        """
        request = {
            "bundle": self.bundle,
            "scale": self.scale,
            "device": self.device,
            "seed": int(seed),
            "submission": self.optimizer_path,
            "cache_dir": cache_dir,
        }
        before = snapshot(self.monitored_roots())
        chan_r, chan_w = os.pipe()
        blob_r, blob_w = os.pipe()
        records, fault = [], None
        try:
            log = open(self.child_log, "a")
            proc = subprocess.Popen(
                [sys.executable, TRAIN_CHILD, "--channel-fd", str(chan_w),
                 "--blob-fd", str(blob_w)],
                stdin=subprocess.PIPE, stdout=log, stderr=log,
                pass_fds=(chan_w, blob_w), cwd=self.cwd, env=self.child_env())
            os.close(chan_w)
            os.close(blob_w)
            chan_w = blob_w = None
            proc.stdin.write(json.dumps(request).encode())
            proc.stdin.close()
            self.reset_measure(seed)
            chan = os.fdopen(chan_r, "r")
            blob = os.fdopen(blob_r, "rb")
            chan_r = blob_r = None
            try:
                for line in chan:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        raise ChildFailure("child-channel-malformed",
                                           {"seed": seed, "line": line[:200]})
                    if obj.get("kind") == "fault":
                        fault = obj
                    elif obj.get("kind") == "checkpoint":
                        payload = _read_exactly(blob, int(obj["blob_bytes"]))
                        on_measured(self.measure_blob(
                            int(obj["seed"]), int(obj["step"]), payload))
                    else:
                        records.append(obj)
            finally:
                chan.close()
                blob.close()
            rc = proc.wait()
        finally:
            for fd in (chan_r, chan_w, blob_r, blob_w):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        after = snapshot(self.monitored_roots())
        self.write_effects.extend(diff_snapshots(before, after))
        if fault is not None:
            raise ChildFailure("child-faulted", {"seed": seed, "fault": fault})
        if rc != 0:
            raise ChildFailure("child-exit-nonzero", {"seed": seed, "returncode": rc})
        if not records:
            raise ChildFailure("child-produced-no-records", {"seed": seed})
        return records

    def write_records(self, records):
        """The parent, not the child, is the writer of the graded record file."""
        with open(self.record_path, "w") as fh:
            for index, rec in enumerate(records, 1):
                rec = dict(rec)
                rec["verifier_record_index"] = index
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return self.record_path
