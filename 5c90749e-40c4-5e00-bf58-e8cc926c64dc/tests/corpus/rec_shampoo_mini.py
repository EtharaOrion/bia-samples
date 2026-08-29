"""Excluded recipe: inverse fourth-root Shampoo on two-dimensional blocks."""
import torch


def _inv_root(M, power, eps=1e-6):
    M = M.float()
    d = M.size(0)
    M = M + eps * torch.eye(d, device=M.device)
    w, Q = torch.linalg.eigh(M)
    w = w.clamp_min(eps).pow(-1.0 / power)
    return (Q * w) @ Q.T


class ShampooMini(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, update_freq=20, eps=1e-6):
        super().__init__(params, dict(lr=lr, momentum=momentum, update_freq=update_freq, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.dim() != 2:
                    continue
                g = p.grad
                st = self.state[p]
                if "L" not in st:
                    st["L"] = torch.zeros(g.size(0), g.size(0), device=g.device)
                    st["R"] = torch.zeros(g.size(1), g.size(1), device=g.device)
                    st["iL"] = torch.eye(g.size(0), device=g.device)
                    st["iR"] = torch.eye(g.size(1), device=g.device)
                    st["buf"] = torch.zeros_like(g)
                    st["k"] = 0
                st["k"] += 1
                st["L"].add_(g.float() @ g.float().T)
                st["R"].add_(g.float().T @ g.float())
                if st["k"] % group["update_freq"] == 1:
                    st["iL"] = _inv_root(st["L"], 4, group["eps"])
                    st["iR"] = _inv_root(st["R"], 4, group["eps"])
                st["buf"].mul_(group["momentum"]).add_(g)
                upd = st["iL"].to(g.dtype) @ st["buf"] @ st["iR"].to(g.dtype)
                p.add_(upd, alpha=-group["lr"])
