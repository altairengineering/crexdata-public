import torch


class FisherMatrix:
    def __init__(self, tracked_params, beta=0.95, eps=1e-8, dtype=torch.float32):
        """
        tracked_params: List[(name: str, param: nn.Parameter)]
        """
        self.beta = float(beta)
        self.eps = float(eps)
        self.dtype = dtype
        self.tracked = list(tracked_params)

        # store as dict of tensors (not buffers, not tied to model)
        self.fisher = {
            name: torch.zeros_like(p, dtype=dtype, device=p.device)
            for name, p in self.tracked
        }

    @torch.no_grad()
    def update_from_grads(self, g2_dict: dict[str, torch.Tensor]):
        """
        g2_dict: name -> grad^2 tensor matching the param's shape
        Performs EMA: F = beta * F + (1-beta) * g^2
        """
        for name, _ in self.tracked:
            g2 = g2_dict.get(name)
            if g2 is None:
                continue
            F = self.fisher[name]
            F.mul_(self.beta).add_((1.0 - self.beta) * g2.to(F.dtype))

    def fisher_diag(self, cast_to=torch.float32) -> dict[str, torch.Tensor]:
        return {name: F.to(cast_to) for name, F in self.fisher.items()}


class FIMLoss:
    def __init__(self, tracked_params, lam=0.1, beta=0.95, eps=1e-8):
        self.fs = FisherMatrix(tracked_params, beta, eps)
        self.lam = float(lam)
        self.eps = float(eps)

    def _weight_batch(self, grad_dict, clamp=None):
        F = self.fs.fisher_diag(cast_to=torch.float32)

        terms = []
        for name, _ in self.fs.tracked:
            g = grad_dict.get(name)
            if g is None:
                continue
            terms.append((g.to(torch.float32).pow(2) / (F[name] + self.eps)).mean())

        if not terms:
            device = self.fs.tracked[0][1].device
            return torch.tensor(1.0, device=device, dtype=torch.float32)

        num = torch.stack(terms).mean()
        w = (1.0 + self.lam * num).to(torch.float32)

        # normalize around 1
        w = w / w.detach().clamp_min(1e-8)
        if clamp is not None:
            w = torch.clamp(w, clamp[0], clamp[1])
        return w

    def _weight_per_sample(self, mse_i, clamp=None):
        # compute scale from inverse Fisher
        Fdiag = self.fs.fisher_diag(cast_to=torch.float32)
        if len(Fdiag) > 0:
            inv_means = [
                (1.0 / (Fdiag[name] + 1e-8)).mean() for (name, _) in self.fs.tracked
            ]
            s = torch.stack(inv_means).mean()
        else:
            s = torch.tensor(0.0, device=mse_i.device, dtype=torch.float32)

        w_i = 1.0 + self.lam * s.to(mse_i.dtype) * mse_i
        w_i = w_i / w_i.mean().clamp_min(1e-8)  # normalize
        if clamp is not None:
            w_i = torch.clamp(w_i, clamp[0], clamp[1])
        return w_i

    def fim_weight(self, grad_dict, mse_i=None, per_sample=True, clamp=None):
        """
        Args:
          grad_dict: dict[name->Tensor] detached grads
          mse_i: optional, (B,) per-sample MSE values
          per_sample: not yet discussed (proxy branch)
        """
        if per_sample:
            return self._weight_per_sample(mse_i, clamp)
        else:
            return self._weight_batch(grad_dict, clamp)
