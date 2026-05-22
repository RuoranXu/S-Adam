import torch
from torch.optim import Optimizer
import math


class SAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2,
                 k_directions=2,
                 sigma=0.01,
                 lgi_lambda=2.0,
                 stabilize_eps=1e-6):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        k_directions=k_directions, sigma=sigma,
                        lgi_lambda=lgi_lambda, stabilize_eps=stabilize_eps)
        super(SAdam, self).__init__(params, defaults)
        self.lgi_history = []

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("S-Adam requires a closure to estimate geometry.")

        loss = None
        with torch.enable_grad():
            loss = closure(backward=True)
        base_loss = loss.item()

        for group in self.param_groups:
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            lgi_score = 0.0
            damping = 1.0

            if group['lgi_lambda'] > 0:
                k = group['k_directions']
                sigma = group['sigma']
                diffs = []

                for _ in range(k):
                    noise_cache = []
                    for p in params_with_grad:
                        u = torch.randn_like(p)
                        u = u / (u.norm() + 1e-12)
                        perturbation = sigma * u
                        p.data.add_(perturbation)
                        noise_cache.append(perturbation)

                    with torch.no_grad():
                        try:
                            loss_perturbed = closure(backward=False)
                        except TypeError:
                            loss_perturbed = closure()

                    d_i = (loss_perturbed.item() - base_loss) / sigma
                    diffs.append(d_i)

                    for p, pert in zip(params_with_grad, noise_cache):
                        p.data.sub_(pert)

                d_tensor = torch.tensor(diffs)
                var_d = torch.var(d_tensor, unbiased=True) if k > 1 else torch.tensor(0.0)
                mean_sq_d = torch.mean(d_tensor ** 2)

                lgi_score = var_d / (mean_sq_d + group['stabilize_eps'])
                lgi_score = lgi_score.item()

                safe_lgi = min(lgi_score, 10.0)
                damping = math.exp(-group['lgi_lambda'] * safe_lgi)

            beta1, beta2 = group['betas']
            for p in params_with_grad:
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size * damping)

        if len(self.param_groups[0]['params']) > 0:
            self.lgi_history.append(lgi_score)

        return loss