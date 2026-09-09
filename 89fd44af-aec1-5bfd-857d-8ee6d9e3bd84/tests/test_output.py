import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from checkers import outcomes

def _o():
    return outcomes.load()

def test_frozen_recipe_unchanged():
    assert _o()['frozen_recipe_unchanged'] is True

def test_one_fwd_bwd_per_step():
    assert _o()['one_fwd_bwd_per_step'] is True

def test_multi_seed_noise_floor():
    assert _o()['multi_seed_noise_floor'] is True

def test_faithful_reporting():
    assert _o()['faithful_reporting'] is True

def test_telemetry_chain_verified():
    assert _o()['telemetry_chain_verified'] is True

def test_submission_binding_verified():
    assert _o()['submission_binding_verified'] is True

def test_full_run_mode_verified():
    assert _o()['full_run_mode_verified'] is True

def test_optimizer_not_verbatim_copy():
    assert _o()['optimizer_not_verbatim_copy'] is True

def test_full_score_target_reached():
    assert _o()['full_score_target_reached'] is True

import pathlib

import importlib.util
import pytest
torch = pytest.importorskip("torch")
SUBMISSION = pathlib.Path(os.environ.get("TRACK3_OPTIMIZER", "/workspace/submission/optimizer.py"))


SUBMISSION = pathlib.Path(
    os.environ.get("TRACK3_OPTIMIZER", "/workspace/submission/optimizer.py")
)


@pytest.fixture(autouse=True, scope="session")
def _process_group():
    import torch.distributed as dist
    if not dist.is_available() or dist.is_initialized():
        yield
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    try:
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    except Exception:
        yield
        return
    yield
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _load():
    if not SUBMISSION.is_file():
        pytest.skip(f"no submission at {SUBMISSION}")
    spec = importlib.util.spec_from_file_location("submitted_optimizer", SUBMISSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _params(n=2, shape=(8, 4), seed=0):
    torch.manual_seed(seed)
    ps = []
    for _ in range(n):
        p = torch.nn.Parameter(torch.randn(*shape))
        p.grad = torch.randn(*shape)
        ps.append(p)
    return ps


def test_build_optimizer_exists():
    assert callable(getattr(_load(), "build_optimizer", None)), \
        "instruction.md requires build_optimizer(params, lr=..., **kwargs)"


def test_returns_torch_optimizer():
    opt = _load().build_optimizer(_params())
    assert isinstance(opt, torch.optim.Optimizer), f"got {type(opt).__name__}"


def test_accepts_bare_tensors_and_named_pairs():
    m = _load()
    m.build_optimizer(_params())
    ps = _params()
    m.build_optimizer([(f"layer.{i}", p) for i, p in enumerate(ps)])


def test_honours_lr_argument():
    m = _load()
    lo = m.build_optimizer(_params(seed=1), lr=1e-4)
    hi = m.build_optimizer(_params(seed=1), lr=1e-1)

    def moved(opt):
        p0 = [g["params"][0].detach().clone() for g in opt.param_groups]
        opt.step()
        return sum((g["params"][0] - a).abs().sum().item()
                   for g, a in zip(opt.param_groups, p0))

    assert moved(hi) > moved(lo), "step size did not respond to lr; the frozen recipe sets it"


def test_step_applies_exactly_one_update():
    m = _load()
    ps = _params(seed=2)
    opt = m.build_optimizer(ps)
    before = ps[0].detach().clone()
    opt.step()
    one = (ps[0] - before).abs().sum().item()
    mid = ps[0].detach().clone()
    opt.step()
    two = (ps[0] - mid).abs().sum().item()
    assert one > 0, "step() did not update the parameter"
    assert two > 0, "second step() was a no-op; one call must be one update"


def test_only_touches_parameters_it_was_given():
    m = _load()
    ps = _params(seed=3)
    outsider = torch.nn.Parameter(torch.randn(8, 4))
    outsider.grad = torch.randn(8, 4)
    keep = outsider.detach().clone()
    m.build_optimizer(ps).step()
    assert torch.equal(outsider, keep), \
        "submission mutated a parameter it does not own; the harness owns everything else"


def test_deterministic_under_fixed_seed():
    m = _load()
    out = []
    for _ in range(2):
        ps = _params(seed=4)
        m.build_optimizer(ps).step()
        out.append(ps[0].detach().clone())
    assert torch.allclose(out[0], out[1], atol=0, rtol=0), \
        "two build_optimizer+step runs from seed 4 differ at atol=0 rtol=0"


def test_step_does_no_io():
    m = _load()
    ps = _params(seed=5)
    opt = m.build_optimizer(ps)
    real_open = open
    seen = []

    def guard(f, mode="r", *a, **k):
        if any(x in str(mode) for x in ("w", "a", "+")):
            seen.append(str(f))
        return real_open(f, mode, *a, **k)

    import builtins
    builtins.open = guard
    try:
        opt.step()
    finally:
        builtins.open = real_open
    assert not seen, f"step() opened {seen} in a write or append mode"
