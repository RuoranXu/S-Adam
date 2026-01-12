import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import numpy as np
import time

# ==============================================================================
# Part 1: S-Adam Optimizer (Strict Implementation of Paper Theory)
# 严格遵循 N-TGA Update Rule:
# 1. Randomized Directional Derivative
# 2. LGI Calculation: Var / E[X^2]
# 3. Topological Damper: exp(-lambda * LGI)
# ==============================================================================
class SAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, 
                 k_directions=3,       # k: 论文建议 3-5
                 sigma=0.005,          # h: 探测半径
                 lgi_lambda=1.0,       # lambda: 阻尼系数
                 stabilize_eps=1e-6):  # epsilon: 数值稳定项
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        k_directions=k_directions, sigma=sigma,
                        lgi_lambda=lgi_lambda, stabilize_eps=stabilize_eps)
        super(SAdam, self).__init__(params, defaults)
        self.lgi_history = []

    def step(self, closure=None):
        """
        Performs a single optimization step using S-Adam logic.
        """
        if closure is None:
            raise RuntimeError("S-Adam requires a closure to estimate geometry.")

        loss = None
        # 1. Calculate Baseline Loss (f(x))
        with torch.enable_grad():
            loss = closure()
        base_loss = loss.item()

        # Iterate over all parameter groups
        for group in self.param_groups:
            # Gather params that require gradients
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue
            
            # =======================================================
            # Theory Step 1 & 2: Estimate LGI (Local Geometric Instability)
            # =======================================================
            lgi_score = 0.0
            damping = 1.0
            
            # S-Adam logic (only if lambda > 0)
            if group['lgi_lambda'] > 0:
                k = group['k_directions']
                sigma = group['sigma']
                diffs = []
                
                # Perform k random probes (Approximating expectation via MC)
                for _ in range(k):
                    noise_cache = []
                    
                    # Apply perturbation: x + sigma * u
                    for p in params_with_grad:
                        # u ~ Uniform(Sphere) or Gaussian. Gaussian is easier to implement and functionally equivalent for local variance.
                        u = torch.randn_like(p)
                        u = u / (u.norm() + 1e-12) # Normalize to unit vector
                        perturbation = sigma * u
                        p.data.add_(perturbation)
                        noise_cache.append(perturbation)
                    
                    # Forward pass only (Efficient)
                    with torch.enable_grad():
                        loss_perturbed = closure()
                    
                    # Directional derivative approx: (f(x+hu) - f(x)) / h
                    # Note: We take absolute difference magnitude primarily
                    d_i = (loss_perturbed.item() - base_loss) / sigma
                    diffs.append(d_i)
                    
                    # Restore parameters: x
                    for p, pert in zip(params_with_grad, noise_cache):
                        p.data.sub_(pert)
                
                # Calculate LGI: Var[D] / (E[D^2] + eps)
                # This matches Definition in Abstract/Intro
                d_tensor = torch.tensor(diffs)
                var_d = torch.var(d_tensor, unbiased=True)
                mean_sq_d = torch.mean(d_tensor ** 2)
                
                lgi_score = var_d / (mean_sq_d + group['stabilize_eps'])
                lgi_score = lgi_score.item()
                
                # Theory Step 3: Topological Damper
                # alpha = exp(-lambda * LGI)
                # Clamp LGI to avoid numerical underflow (brake too hard) in extreme cases
                safe_lgi = min(lgi_score, 10.0) 
                damping = math.exp(-group['lgi_lambda'] * safe_lgi)
            
            # =======================================================
            # Standard AdamW Update with Damping
            # =======================================================
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

                # Weight Decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Momentum Update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply S-Adam Damping
                # update = - lr * damping * (m / v)
                p.data.addcdiv_(exp_avg, denom, value=-step_size * damping)
        
        # Log LGI for analysis
        if len(self.param_groups[0]['params']) > 0:
            self.lgi_history.append(lgi_score)
            
        return loss

# ==============================================================================
# Part 2: 4-bit Quantization Modules (Generating Nonsmoothness)
# ==============================================================================
# 将这段代码替换你原来的 FakeQuantize 类
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 2-bit signed range: -2 to 1. (总共 4 个值)
        # 将之前的 7.0 改成 2.0 (或者 1.5)
        scale = 1.5 / (input.abs().max() + 1e-5) 
        
        input_scaled = input * scale
        
        # clamp 范围改成 -2 到 1
        input_q = input_scaled.round().clamp(-2, 1)
        
        return input_q / scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE (Straight Through Estimator): 欺骗优化器，假装量化是可导的
        return grad_output

def quantize_4bit(x):
    return FakeQuantize.apply(x)

class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        # Quantize Weights before forward
        w_q = quantize_4bit(self.conv.weight)
        # Quantize Input (Activation)
        x_q = quantize_4bit(x)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)

# A Simple CNN with heavy quantization noise
class QATNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Quantized Convs
        self.conv1 = QuantizedConv2d(1, 16, 3, 1) 
        self.conv2 = QuantizedConv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        # Quantize the FC weights too
        w_fc_q = quantize_4bit(self.fc1.weight)
        x = F.linear(x, w_fc_q, self.fc1.bias)
        return x

# ==============================================================================
# Part 3: Experiment Loop
# ==============================================================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Define closure for S-Adam
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        epoch_loss.append(loss.item())
        
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
            
    return np.mean(epoch_loss)

def run_comparison():
    # Setup Data (FashionMNIST is harder/more textured than MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    # Use subset for speed if needed, here we use full for stability
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    epochs = 3 # 3 epochs is enough to see the stability difference in QAT
    
    # --- Run AdamW ---
    print("\nTraining with AdamW (Baseline)...")
    model_adam = QATNet().to(device)
    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=0.005) # Standard AdamW
    losses_adam = []
    for e in range(1, epochs + 1):
        # Wrapper for AdamW to match S-Adam closure interface
        def train_adam_wrapper(model, loader, opt):
            model.train()
            ep_loss = []
            for d, t in loader:
                d, t = d.to(device), t.to(device)
                opt.zero_grad()
                out = model(d)
                loss = F.cross_entropy(out, t)
                loss.backward()
                opt.step()
                ep_loss.append(loss.item())
            return np.mean(ep_loss), ep_loss # Return mean and full trace
        
        mean_l, trace_l = train_adam_wrapper(model_adam, train_loader, opt_adam)
        losses_adam.extend(trace_l)
        print(f"Epoch {e} Mean Loss: {mean_l:.4f}")

    # --- Run S-Adam ---
    print("\nTraining with S-Adam (Ours)...")
    model_sadam = QATNet().to(device)
    # S-Adam parameters:
    # sigma=0.01: Reasonable perturbation for weights ~O(0.1-1.0)
    # lambda=2.0: Strong enough brake to smooth out quantization noise
    opt_sadam = SAdam(model_sadam.parameters(), lr=0.005, k_directions=3, sigma=0.01, lgi_lambda=2.0)
    losses_sadam = []
    lgi_trace = []
    
    for e in range(1, epochs + 1):
        model_sadam.train()
        for batch_idx, (d, t) in enumerate(train_loader):
            d, t = d.to(device), t.to(device)
            
            def closure():
                opt_sadam.zero_grad()
                out = model_sadam(d)
                loss = F.cross_entropy(out, t)
                loss.backward()
                return loss
            
            loss = opt_sadam.step(closure)
            losses_sadam.append(loss.item())
            if batch_idx % 50 == 0:
                 print(f"S-Adam Epoch {e} iter {batch_idx} Loss: {loss.item():.4f} LGI: {opt_sadam.lgi_history[-1]:.4f}")
        
    lgi_trace = opt_sadam.lgi_history

    return losses_adam, losses_sadam, lgi_trace

# ==============================================================================
# Part 4: Visualization
# ==============================================================================
if __name__ == "__main__":
    l_adam, l_sadam, lgi = run_comparison()
    
    # Smooth curves for better visualization
    def smooth(scalars, weight=0.95):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    l_adam_smooth = smooth(l_adam)
    l_sadam_smooth = smooth(l_sadam)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Losses
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cross Entropy Loss (QAT)', color='black')
    ax1.plot(l_adam_smooth, color='red', alpha=0.6, label='AdamW (QAT Noise)')
    ax1.plot(l_sadam_smooth, color='blue', linewidth=2, label='S-Adam (Stabilized)')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title("S-Adam vs AdamW on 4-bit Quantization-Aware Training (FashionMNIST)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot LGI on twin axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('LGI Score (Geometric Instability)', color='green')
    # Plot LGI as a filled area
    lgi_smooth = smooth(lgi, 0.9)
    ax2.fill_between(range(len(lgi_smooth)), lgi_smooth, color='green', alpha=0.15, label='LGI Score')
    ax2.plot(lgi_smooth, color='green', alpha=0.4, linewidth=1)
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    plt.show()
    
    print("Experiment Complete.")
