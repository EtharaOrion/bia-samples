"""Live normalization probe.

This is the instrument that makes the ablation enforced rather than requested.
It records, while the run is happening, every normalization operation that the
process actually executed, tagged with the phase it executed in. A submission
cannot quietly reintroduce activation normalization into the graded forward
pass, because doing so writes an event into the trace the checker reads.

Two independent interception points are installed so a single bypass does not
blind the probe:

1. Every torch.nn.functional normalization entry point is wrapped, which catches
   a direct functional call from anywhere in the process, including from inside
   the submitted update rule.
2. A forward pre-hook is registered on every module of the live model, which
   records the class name of each module actually executed and catches a
   normalization module smuggled into the module tree at runtime.

The probe also records the distinct set of module classes executed per phase.
That set is small, so the trace stays tiny even over a full run, and it is the
positive half of the read: a checker can confirm the forward pass really ran
the modules it was supposed to run rather than confirm only that nothing bad
appeared.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import config as C


class NormProbe:
    def __init__(self, runlog=None):
        self.runlog = runlog
        self.phase = "init"
        self.events = []
        self.module_classes = {}
        self.module_forward_counts = {}
        self._originals = {}
        self._handles = []
        self._installed = False

    def set_phase(self, phase: str):
        self.phase = phase

    def _record_event(self, op: str, kind: str, detail=None):
        rec = {"phase": self.phase, "op": op, "kind": kind}
        if detail is not None:
            rec["detail"] = detail
        self.events.append(rec)
        if self.runlog is not None:
            self.runlog.emit("norm_event", phase=self.phase, op=op, kind=kind)

    def install_functional_guards(self):
        if self._installed:
            return
        for name in C.ABLATED_FUNCTIONALS:
            orig = getattr(F, name, None)
            if orig is None:
                continue
            self._originals[name] = orig

            def make(nm, fn):
                def wrapper(*args, **kwargs):
                    probe._record_event(f"torch.nn.functional.{nm}", "functional")
                    return fn(*args, **kwargs)

                return wrapper

            probe = self
            setattr(F, name, make(name, orig))
            setattr(torch.nn.functional, name, getattr(F, name))
        self._installed = True

    def remove_functional_guards(self):
        for name, orig in self._originals.items():
            setattr(F, name, orig)
            setattr(torch.nn.functional, name, orig)
        self._originals = {}
        self._installed = False

    def hook_model(self, model):
        def pre_hook(mod, _inputs):
            cls = type(mod).__name__
            phase = self.phase
            seen = self.module_classes.setdefault(phase, set())
            seen.add(cls)
            self.module_forward_counts[phase] = self.module_forward_counts.get(phase, 0) + 1
            if cls in C.ABLATED_MODULE_CLASSES:
                self._record_event(cls, "module")

        for _, mod in model.named_modules():
            self._handles.append(mod.register_forward_pre_hook(pre_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def trace(self):
        return {
            "events": list(self.events),
            "module_classes_by_phase": {
                k: sorted(v) for k, v in sorted(self.module_classes.items())
            },
            "module_forward_counts_by_phase": dict(sorted(self.module_forward_counts.items())),
            "forbidden_phases": list(C.FORBIDDEN_PHASES),
            "watched_module_classes": list(C.ABLATED_MODULE_CLASSES),
            "watched_functionals": list(C.ABLATED_FUNCTIONALS),
        }

    def forbidden_events(self):
        return [e for e in self.events if e["phase"] in C.FORBIDDEN_PHASES]
