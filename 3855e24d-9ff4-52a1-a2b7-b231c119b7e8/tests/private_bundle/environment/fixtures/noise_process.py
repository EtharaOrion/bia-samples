"""Frozen injected gradient noise process for bia slot S06.

This file is a FROZEN FIXTURE. It is part of the task substrate, exactly like the
model architecture and the corpus. Editing it, reseeding it, monkeypatching it or
bypassing it invalidates the run, because the verifier recomputes the realization
this fixture produces from a pristine copy and compares it against what the run
recorded.

The process is deterministic. Given the frozen seed below, the parameter tensor
identity and the step index, the corruption applied to a true gradient is fully
determined, so two graders running the same submission see the same noise
realization. That reproducibility is what makes steps to target measurable at all.

Corruption model, applied per parameter tensor, in this exact order:

  1. anisotropic amplification   x = g + BETA * dot(g, u) * u
  2. block cyclic erasure        x = x * m_t / (1 - P_DROP)
  3. correlated additive drift   x = x + RHO * s_t * norm(g) * d_t
  4. scheduled heavy spike       x = x + KAPPA * norm(g) * q_t on frozen spike steps

Every component scales with the true gradient, so the corruption vanishes at a
stationary point and the optimum of the frozen objective is unchanged. What the
components destroy is the descent direction on the way there.

Component notes, stated plainly because the fixture is public. The vector u is one
frozen unit direction per tensor, so the amplification is neither zero mean nor
independent of the gradient and averaging observations across steps does not
remove it, which is why it survives momentum, exponential moving averages and the
Adam second moment alike. The mask m_t erases four of sixteen frozen coordinate
blocks and holds the same erasure for BLOCK_HOLD consecutive steps, so erased
coordinates go stale in runs rather than independently per step, and the
1/(1 - P_DROP) rescale restores an unbiased estimator only under an independence
assumption this schedule denies. The scalar s_t is a frozen sequence with
correlation time DRIFT_TAU, and positive autocorrelation over tens of steps is
amplified by a short window average instead of cancelled by it. The vector q_t
fires on a frozen sparse schedule with a large multiplier, so a single observation
can be dominated by a term that carries no descent information at all.

Erasure is lossy. No single observation determines the true gradient, so the
process is not invertible one step at a time whatever a reader knows about it.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

# --- frozen constants. Changing any byte here changes the task. ---------------
NOISE_SEED = 20260821
BETA = 3.0
P_DROP = 0.25
N_BLOCKS = 16
N_DROPPED = 4
BLOCK_HOLD = 4
BLOCK_STRIDE = 5
RHO = 2.75
DRIFT_TAU = 64.0
KAPPA = 12.0
SPIKE_PERIOD = 23
SPIKE_PHASE = 7
DRIFT_OMEGA_PERIOD = 97.0
PROBE_NUMEL = 512
FIXTURE_VERSION = "s06-noise-1.0.0"

_DRIFT_MEMO = [0.0]
_DRIFT_INNOV = None


def _key(label: str) -> int:
    """Deterministic 64 bit counter based key from the frozen seed and a label."""
    material = ("%d|%s" % (NOISE_SEED, label)).encode("ascii")
    return struct.unpack("<Q", hashlib.sha256(material).digest()[:8])[0]


def _stream(label: str, count: int) -> np.ndarray:
    """Deterministic float64 stream in [-1, 1) of the requested length.

    Built from raw Philox counter output rather than from a distribution method,
    because raw bit generator output is the part numpy guarantees reproducible.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    raw = np.random.Philox(_key(label)).random_raw(int(count)).astype(np.uint64)
    return (raw >> np.uint64(1)).astype(np.float64) / float(1 << 63) * 2.0 - 1.0


def _perm(label: str, n: int) -> np.ndarray:
    """Deterministic permutation of range(n)."""
    return np.argsort(_stream(label + "|perm", n), kind="stable").astype(np.int64)


def spike_step(t: int) -> bool:
    """Frozen sparse spike schedule. Public, deterministic, step indexed."""
    return (int(t) % SPIKE_PERIOD) == SPIKE_PHASE


def dropped_blocks(t: int):
    """Frozen block erasure schedule, held for BLOCK_HOLD consecutive steps."""
    phase = int(t) // BLOCK_HOLD
    return tuple(sorted(((phase * BLOCK_STRIDE + j) % N_BLOCKS) for j in range(N_DROPPED)))


def drift_scalar(t: int) -> float:
    """Frozen scalar sequence with correlation time DRIFT_TAU, mean zero over a long run."""
    global _DRIFT_INNOV
    t = int(t)
    if t < len(_DRIFT_MEMO):
        return _DRIFT_MEMO[t]
    need = max(t + 1, 4096)
    if _DRIFT_INNOV is None or len(_DRIFT_INNOV) < need:
        _DRIFT_INNOV = _stream("drift", need)
    a = math.exp(-1.0 / DRIFT_TAU)
    scale = math.sqrt(max(1.0 - a * a, 1e-12))
    s = _DRIFT_MEMO[-1]
    for k in range(len(_DRIFT_MEMO), t + 1):
        s = a * s + scale * float(_DRIFT_INNOV[k])
        _DRIFT_MEMO.append(s)
    return _DRIFT_MEMO[t]


def drift_mix(t: int):
    """Frozen rotation coefficients for the drift direction at step t."""
    omega = 2.0 * math.pi / DRIFT_OMEGA_PERIOD
    return math.cos(omega * int(t)), math.sin(omega * int(t))


class TensorNoise:
    """Per tensor frozen noise state, built once from the tensor identity and size."""

    def __init__(self, torch, tensor_id: str, numel: int, device):
        self.torch = torch
        self.tensor_id = tensor_id
        self.numel = int(numel)
        label = "%s|%d" % (tensor_id, self.numel)

        def unit(name):
            v = torch.as_tensor(_stream(label + "|" + name, self.numel), dtype=torch.float32)
            v = v.to(device)
            return v / (v.norm() + 1e-12)

        self.u = unit("u")
        self.d0 = unit("d0")
        d1 = unit("d1")
        d1 = d1 - (d1 @ self.d0) * self.d0
        self.d1 = d1 / (d1.norm() + 1e-12)
        self.q = unit("q")
        pos = torch.as_tensor(_perm(label + "|blocks", self.numel), dtype=torch.int64).to(device)
        lane = (torch.arange(self.numel, device=device, dtype=torch.int64) * N_BLOCKS) // max(self.numel, 1)
        assign = torch.zeros(self.numel, dtype=torch.int64, device=device)
        assign[pos] = lane
        self.block_of = assign
        self._mask_key = None
        self._mask = None

    def mask(self, t: int):
        """Frozen keep mask at step t, one entry per coordinate."""
        key = int(t) // BLOCK_HOLD
        if key == self._mask_key and self._mask is not None:
            return self._mask
        torch = self.torch
        keep = torch.ones(self.numel, dtype=torch.float32, device=self.block_of.device)
        for b in dropped_blocks(t):
            keep = keep * (self.block_of != b).to(torch.float32)
        self._mask_key = key
        self._mask = keep
        return keep

    def drift_dir(self, t: int):
        """Frozen unit drift direction at step t."""
        c, s = drift_mix(t)
        v = c * self.d0 + s * self.d1
        return v / (v.norm() + 1e-12)


class NoiseProcess:
    """Frozen corruption applied to true gradients before any estimator sees them."""

    def __init__(self, torch):
        self.torch = torch
        self._per_tensor = {}

    def state(self, tensor_id, numel, device):
        key = (tensor_id, int(numel))
        st = self._per_tensor.get(key)
        if st is None:
            st = TensorNoise(self.torch, tensor_id, int(numel), device)
            self._per_tensor[key] = st
        return st

    def corrupt(self, tensor_id: str, grad, step: int):
        """Return the corrupted observation of one true gradient tensor at one step."""
        torch = self.torch
        flat = grad.detach().reshape(-1).to(torch.float32)
        st = self.state(tensor_id, flat.numel(), flat.device)
        gnorm = float(flat.norm())
        x = flat + BETA * (flat @ st.u) * st.u
        x = x * st.mask(step) / (1.0 - P_DROP)
        x = x + (RHO * drift_scalar(step) * gnorm) * st.drift_dir(step)
        if spike_step(step):
            x = x + (KAPPA * gnorm) * st.q
        return x.reshape(grad.shape)

    def probe_digest(self, steps) -> str:
        """Reproducible digest of the realization on a synthetic probe tensor.

        The verifier recomputes this from a pristine copy of this fixture and
        compares it against the digest the run recorded, so an edited, reseeded
        or bypassed fixture produces a different digest.
        """
        torch = self.torch
        g = torch.as_tensor(_stream("probe|g", PROBE_NUMEL), dtype=torch.float32, device="cpu")
        h = hashlib.sha256()
        h.update(FIXTURE_VERSION.encode("ascii"))
        for t in steps:
            out = self.corrupt("__probe__", g, int(t))
            vals = out.reshape(-1)[:64].tolist()
            h.update(b"|".join(("%.4e" % float(v)).encode("ascii") for v in vals))
        return h.hexdigest()


def fixture_digest(path: str = None) -> str:
    """sha256 over this fixture's own bytes, so a run can record what it loaded."""
    target = path or __file__
    with open(target, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
