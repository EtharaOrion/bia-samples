"""Excluded recipe: Shampoo-style preconditioning in an Adam eigenbasis."""
import torch


class SoapLight(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, precondition_frequency=10):
        super().__init__(params, dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta,
                                      eps=eps, precondition_frequency=precondition_frequency))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None or p.dim() < 2:
                    continue
                g = p.grad
                st = self.state[p]
                if "L" not in st:
                    st["L"] = torch.zeros(g.size(0), g.size(0), device=g.device, dtype=g.dtype)
                    st["R"] = torch.zeros(g.size(1), g.size(1), device=g.device, dtype=g.dtype)
                    st["QL"] = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
                    st["QR"] = torch.eye(g.size(1), device=g.device, dtype=g.dtype)
                    st["m"] = torch.zeros_like(g)
                    st["v"] = torch.zeros_like(g)
                    st["k"] = 0
                st["k"] += 1
                sb = group["shampoo_beta"]
                st["L"].mul_(sb).add_(g @ g.T, alpha=1 - sb)
                st["R"].mul_(sb).add_(g.T @ g, alpha=1 - sb)
                if st["k"] % group["precondition_frequency"] == 0:
                    st["QL"] = torch.linalg.eigh(st["L"].float())[1].to(g.dtype)
                    st["QR"] = torch.linalg.eigh(st["R"].float())[1].to(g.dtype)
                gr = st["QL"].T @ g @ st["QR"]
                st["m"].mul_(b1).add_(gr, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                upd = st["m"] / (st["v"].sqrt() + group["eps"])
                p.add_(st["QL"] @ upd @ st["QR"].T, alpha=-group["lr"])
