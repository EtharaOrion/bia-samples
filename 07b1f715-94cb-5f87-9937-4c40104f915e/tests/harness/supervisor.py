"""Parent side of the S02 verifier's isolated re-execution.

The parent owns the private root, the record file, the checkpoint directory and
the measurement. The training child streams one record per evaluation over a
pipe the parent created; the parent hands each checkpoint to a long-lived
measurement process that holds no submitted byte, records the loss that process
returns, and deletes the checkpoint behind it, so peak disk stays at a couple
of checkpoints rather than at the whole run.

There is no shared secret anywhere in this design. The previous revision
chained the record under a key the solving agent had to hold in order to write
at all, which is not a signature. The record is now written by a process the
agent cannot reach, so there is nothing for a key to defend.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
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


def diff_snapshots(before, after):
    changed = []
    for path, digest in sorted(after.items()):
        if path not in before:
            changed.append("created:" + path)
        elif before[path] != digest:
            changed.append("modified:" + path)
    for path in sorted(before):
        if path not in after:
            changed.append("removed:" + path)
    return changed


class Supervisor:
    def __init__(self, bundle, profile, submission_dir, submission_path):
        self.bundle = bundle
        self.profile = profile
        self.submission_dir = os.path.abspath(submission_dir)
        self.submission_path = os.path.abspath(submission_path)
        self.root = tempfile.mkdtemp(prefix="bia-s02-verify-")
        self.scratch = os.path.join(self.root, "scratch")
        self.home = os.path.join(self.root, "home")
        self.cwd = os.path.join(self.root, "cwd")
        for d in (self.scratch, self.home, self.cwd):
            os.makedirs(d, exist_ok=True)
        self.record_path = os.path.join(self.root, "verifier_record.jsonl")
        self.child_log = os.path.join(self.root, "child_stdio.log")
        self.write_effects = []
        # The parent chains its own record under a key it mints here and never
        # sends anywhere. It authenticates the file the parent wrote against
        # the parent that wrote it, which is the only claim a chain can carry
        # once the writer is no longer reachable by the graded party.
        self._chain_key = os.urandom(32)
        self._records = []
        self._measure = None
        self._measure_chan = None

    def child_env(self):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = self.scratch
        env["TMP"] = self.scratch
        env["TEMP"] = self.scratch
        env["HOME"] = self.home
        env["XDG_CACHE_HOME"] = os.path.join(self.home, ".cache")
        env["BIA_GUARD_ALLOW"] = os.pathsep.join((self.scratch, self.home))
        env.pop("BIA_TELEMETRY", None)
        env.pop("BIA_TELEMETRY_DIR", None)
        env.pop("BIA_CHAIN_KEY", None)
        env.pop("BIA_SCORE_PATH", None)
        env.pop("BIA_OUTCOMES", None)
        return env

    def monitored_roots(self):
        return [self.cwd, self.submission_dir]

    def start_measure(self, request):
        read_fd, write_fd = os.pipe()
        log = open(self.child_log, "a")
        proc = subprocess.Popen(
            [sys.executable, MEASURE_CHILD, "--channel-fd", str(write_fd)],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            pass_fds=(write_fd,), cwd=self.cwd, env=self.child_env())
        os.close(write_fd)
        self._measure = proc
        self._measure_chan = os.fdopen(read_fd, "r")
        proc.stdin.write((json.dumps(request) + "\n").encode())
        proc.stdin.flush()
        hello = self._measure_chan.readline()
        if not hello:
            raise ChildFailure("measure-child-did-not-start", {})
        doc = json.loads(hello)
        if doc.get("kind") != "ready":
            raise ChildFailure("measure-child-fault", doc)
        return doc

    def measure_blob(self, payload):
        self._measure.stdin.write(
            (json.dumps({"blob_bytes": len(payload)}) + "\n").encode())
        self._measure.stdin.write(payload)
        self._measure.stdin.flush()
        line = self._measure_chan.readline()
        if not line:
            raise ChildFailure("measure-child-closed-channel", {})
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
            self._measure.wait(timeout=60)
        except Exception:
            self._measure.kill()
        try:
            self._measure_chan.close()
        except Exception:
            pass
        self._measure = None

    def run_seed(self, seed, on_record):
        """Trains one seed. Parameters never touch disk.

        The training child announces each evaluation on the record channel and
        writes its parameters to a second pipe this process owns. The parent
        reads exactly the announced number of bytes and hands them straight to
        the measurement process, so the weights of a step are consumed before
        the training process has produced anything it could overwrite them with.
        """
        request = {
            "bundle": self.bundle,
            "profile": self.profile,
            "seed": int(seed),
            "submission": self.submission_path,
        }
        before = snapshot(self.monitored_roots())
        chan_r, chan_w = os.pipe()
        blob_r, blob_w = os.pipe()
        fault = None
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
            proc.stdin.write((json.dumps(request) + "\n").encode())
            proc.stdin.close()
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
                        continue
                    payload = _read_exactly(blob, int(obj["blob_bytes"]))
                    on_record(obj, payload)
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

    def append_record(self, body):
        """The parent assigns the index and the link. The child does neither."""
        index = len(self._records) + 1
        record = dict(body)
        record["verifier_record_index"] = index
        prev = self._records[-1]["chain"] if self._records else "genesis"
        payload = prev + json.dumps(record, sort_keys=True)
        record["chain"] = hmac.new(self._chain_key, payload.encode(),
                                   hashlib.sha256).hexdigest()
        self._records.append(record)
        return record

    def records(self):
        return list(self._records)

    def verify_own_chain(self):
        prev = "genesis"
        for index, record in enumerate(self._records, 1):
            if record.get("verifier_record_index") != index:
                return False, "record_index_out_of_order_%s" % record.get(
                    "verifier_record_index")
            body = {k: v for k, v in record.items() if k != "chain"}
            payload = prev + json.dumps(body, sort_keys=True)
            expect = hmac.new(self._chain_key, payload.encode(),
                              hashlib.sha256).hexdigest()
            if record.get("chain") != expect:
                return False, "chain_break_at_index_%s" % index
            prev = expect
        return True, None

    def flush_records(self):
        with open(self.record_path, "w") as fh:
            for record in self._records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return self.record_path
