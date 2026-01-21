import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import zipfile
import urllib.request
from PIL import Image


class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# TinyImageNet Dataset
# ==============================================================================
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.join(root, 'tiny-imagenet-200')
        self.train = train
        self.transform = transform

        if download:
            self._download(root)

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = []
        self.targets = []

        # Load wnids (class ids)
        wnids_path = os.path.join(self.root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.classes = [x.strip() for x in f.readlines()]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if self.train:
            # Load training data
            train_dir = os.path.join(self.root, 'train')
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                cls_idx = self.class_to_idx[cls]
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.targets.append(cls_idx)
        else:
            # Load validation data
            val_dir = os.path.join(self.root, 'val')
            img_dir = os.path.join(val_dir, 'images')
            annot_file = os.path.join(val_dir, 'val_annotations.txt')

            with open(annot_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    cls_name = parts[1]

                    if cls_name in self.class_to_idx:
                        self.data.append(os.path.join(img_dir, img_name))
                        self.targets.append(self.class_to_idx[cls_name])

    def _download(self, root):
        if os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            print("Dataset already exists.")
            return

        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        filename = "tiny-imagenet-200.zip"
        filepath = os.path.join(root, filename)

        if not os.path.exists(filepath):
            print(f"Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)

        print("Extracting dataset...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(root)
        print("Done.")

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


# ==============================================================================
# Part 1: S-Adam Optimizer (Ours / Target Method)
# Logic: Randomized Smoothing Probes -> LGI Score -> Adaptive Damping
# ==============================================================================
class SAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2,
                 k_directions=3,
                 sigma=0.005,
                 lgi_lambda=0.1,
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
        # 1. Baseline Loss (Gradients calculated here)
        with torch.enable_grad():
            loss = closure(backward=True)
        base_loss = loss.item()

        for group in self.param_groups:
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            # --- Estimate LGI (Local Geometric Instability) ---
            lgi_score = 0.0
            damping = 1.0

            if group['lgi_lambda'] > 0:
                k = group['k_directions']
                sigma = group['sigma']
                diffs = []

                # k random probes
                for _ in range(k):
                    noise_cache = []
                    # Perturb
                    for p in params_with_grad:
                        u = torch.randn_like(p)
                        u = u / (u.norm() + 1e-12)
                        perturbation = sigma * u
                        p.data.add_(perturbation)
                        noise_cache.append(perturbation)

                    # Forward Probe
                    with torch.no_grad():
                        try:
                            loss_perturbed = closure(backward=False)
                        except TypeError:
                            # Fallback if closure doesn't support backward arg
                            loss_perturbed = closure()

                    # Directional Derivative Approx
                    d_i = (loss_perturbed.item() - base_loss) / sigma
                    diffs.append(d_i)

                    # Restore
                    for p, pert in zip(params_with_grad, noise_cache):
                        p.data.sub_(pert)

                # Calculate Variance / Expectation
                d_tensor = torch.tensor(diffs)
                var_d = torch.var(d_tensor, unbiased=True) if k > 1 else torch.tensor(0.0)
                mean_sq_d = torch.mean(d_tensor ** 2)

                lgi_score = var_d / (mean_sq_d + group['stabilize_eps'])
                lgi_score = lgi_score.item()

                # Topological Damping
                safe_lgi = min(lgi_score, 10.0)
                damping = math.exp(-group['lgi_lambda'] * safe_lgi)

            # --- Standard AdamW Update with Damping ---
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

                # Momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Apply S-Adam Damping
                p.data.addcdiv_(exp_avg, denom, value=-step_size * damping)

        if len(self.param_groups[0]['params']) > 0:
            self.lgi_history.append(lgi_score)

        return loss


# ==============================================================================
# Part 2: Prox-SGD Optimizer (Baseline)
# ==============================================================================
class ProxSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, l1_lambda=1e-4):
        defaults = dict(lr=lr, momentum=momentum, l1_lambda=l1_lambda)
        super(ProxSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            l1_lambda = group['l1_lambda']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # Step 1: Gradient Step
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                p.data.add_(buf, alpha=-lr)

                # Step 2: Proximal Step (Soft Thresholding)
                if l1_lambda > 0:
                    threshold = lr * l1_lambda
                    p_data_abs = p.data.abs()
                    p_data_sign = p.data.sign()
                    p_data_prox = torch.clamp(p_data_abs - threshold, min=0)
                    p.data.copy_(p_data_sign * p_data_prox)

        return loss


# ==============================================================================
# Part 3: Quantization Modules
# ==============================================================================
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 4-bit simulation
        scale = 7.0 / (input.abs().max() + 1e-5)
        input_scaled = input * scale
        input_q = input_scaled.round().clamp(-8, 7)
        return input_q / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_4bit(x):
    return FakeQuantize.apply(x)


class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        w_q = quantize_4bit(self.conv.weight)
        x_q = quantize_4bit(x)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)


class QATNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TinyImageNet is 64x64
        self.conv1 = QuantizedConv2d(3, 16, 3, 1)  # 64 -> 62
        self.conv2 = QuantizedConv2d(16, 32, 3, 1)  # 31 -> 29
        self.conv3 = QuantizedConv2d(32, 64, 3, 1)  # 14 -> 12
        # MaxPool 2x2 after conv1 -> 31
        # MaxPool 2x2 after conv2 -> 14
        # MaxPool 2x2 after conv3 -> 6
        # Feature map size: 64 channels * 6 * 6
        self.fc1 = nn.Linear(64 * 6 * 6, 200)  # 200 classes
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        w_fc_q = quantize_4bit(self.fc1.weight)
        x = F.linear(x, w_fc_q, self.fc1.bias)
        return x


# ==============================================================================
# Part 4: Experiment Loop
# ==============================================================================
def train(model, device, train_loader, optimizer):
    model.train()
    epoch_loss = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss, acc


# ==============================================================================
# Part 5: Synthetic Chattering Demo (Added)
# ==============================================================================
def objective_function(p):
    # Modified Rosenbrock-like Function (Nonsmooth)
    x = p[0]
    y = p[1]
    return 10 * torch.abs(y - x**2) + torch.abs(1 - x)

def run_synthetic_optimizer(opt_class, steps=100, **kwargs):
    # Start at (-1.5, 2.0) - standard challenging start for Rosenbrock
    params = torch.tensor([-1.5, 2.0], requires_grad=True)

    if opt_class == AdamW:
        optimizer = opt_class([params], **kwargs)
    elif opt_class == ProxSGD:
        optimizer = opt_class([params], **kwargs)
    elif opt_class == SAdam:
        optimizer = opt_class([params], **kwargs)

    trajectory = []
    trajectory.append(params.detach().cpu().numpy().copy())

    for _ in range(steps):
        def closure(backward=True):
            if backward:
                optimizer.zero_grad()
            loss = objective_function(params)
            if backward:
                loss.backward()
            return loss

        if isinstance(optimizer, SAdam):
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = objective_function(params)
            loss.backward()
            optimizer.step()

        trajectory.append(params.detach().cpu().numpy().copy())

    return np.array(trajectory)


def run_comparison():
    # Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading TinyImageNet...")
    dataset = TinyImageNet('./data', train=True, download=True, transform=transform)
    test_dataset = TinyImageNet('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 10

    # --- Run Prox-SGD (Baseline) ---
    print("\nTraining with Prox-SGD (Baseline)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    model_prox = QATNet().to(device)
    opt_prox = ProxSGD(model_prox.parameters(), lr=0.01, momentum=0.9, l1_lambda=1e-4)

    losses_prox = []

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time_prox = time.time()

    for e in range(1, epochs + 1):
        model_prox.train()
        ep_loss = []
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)

            def closure():
                opt_prox.zero_grad()
                out = model_prox(d)
                loss = F.cross_entropy(out, t)
                loss.backward()
                return loss

            loss = opt_prox.step(closure)
            ep_loss.append(loss.item())

        losses_prox.extend(ep_loss)
        _, test_acc = test(model_prox, device, test_loader)
        print(f"Prox-SGD Epoch {e} Test Acc: {test_acc:.2f}%")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time_prox = time.time() - start_time_prox
    memory_prox = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # --- Run AdamW (Standard Baseline) ---
    print("\nTraining with AdamW...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    model_adamw = QATNet().to(device)
    opt_adamw = AdamW(model_adamw.parameters(), lr=0.001, weight_decay=1e-2)

    losses_adamw = []

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time_adamw = time.time()

    for e in range(1, epochs + 1):
        model_adamw.train()
        ep_loss = []
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            opt_adamw.zero_grad()
            out = model_adamw(d)
            loss = F.cross_entropy(out, t)
            loss.backward()
            opt_adamw.step()
            ep_loss.append(loss.item())

        losses_adamw.extend(ep_loss)
        _, test_acc = test(model_adamw, device, test_loader)
        print(f"AdamW Epoch {e} Test Acc: {test_acc:.2f}%")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time_adamw = time.time() - start_time_adamw
    memory_adamw = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # --- Run S-Adam (Ours) ---
    print("\nTraining with S-Adam (Ours)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    model_sadam = QATNet().to(device)

    opt_sadam = SAdam(model_sadam.parameters(),
                      lr=0.001,
                      k_directions=8,
                      sigma=0.01,
                      lgi_lambda=2.0)

    losses_sadam = []

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time_sadam = time.time()

    for e in range(1, epochs + 1):
        model_sadam.train()
        for batch_idx, (d, t) in enumerate(train_loader):
            d, t = d.to(device), t.to(device)

            # Optimized closure with backward flag
            def closure(backward=True):
                if backward:
                    opt_sadam.zero_grad()
                out = model_sadam(d)
                loss = F.cross_entropy(out, t)
                if backward:
                    loss.backward()
                return loss

            loss = opt_sadam.step(closure)
            losses_sadam.append(loss.item())

            if batch_idx % 100 == 0:
                lgi = opt_sadam.lgi_history[-1] if opt_sadam.lgi_history else 0
                print(f"S-Adam Ep {e} It {batch_idx} LGI: {lgi:.4f}")

        _, test_acc = test(model_sadam, device, test_loader)
        print(f"S-Adam Epoch {e} Test Acc: {test_acc:.2f}%")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time_sadam = time.time() - start_time_sadam
    memory_sadam = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # === FINAL PRINT RESULT ===
    print("\n" + "=" * 65)
    print("      FINAL PERFORMANCE SUMMARY      ")
    print("=" * 65)
    print(f"{'Method':<20} | {'Total Time (s)':<15} | {'Max Memory (MB)':<15}")
    print("-" * 65)
    print(f"{'Prox-SGD (Base)':<20} | {total_time_prox:.4f}          | {memory_prox:.2f}")
    print(f"{'AdamW (Base)':<20} | {total_time_adamw:.4f}          | {memory_adamw:.2f}")
    print(f"{'S-Adam (Ours)':<20} | {total_time_sadam:.4f}          | {memory_sadam:.2f}")
    print("-" * 65)

    # Calculate Fairness Factor
    ratio = total_time_sadam / total_time_prox
    print(f"Cost Factor: S-Adam is {ratio:.2f}x slower than Prox-SGD.")
    print("=" * 65 + "\n")

    return losses_prox, losses_adamw, losses_sadam, opt_sadam.lgi_history


# ==============================================================================
# Visualization
# ==============================================================================
if __name__ == "__main__":
    sys.stdout = Logger('training_log_prox_adamw_sadam_tinyimagenet1.txt')

    l_prox, l_adamw, l_sadam, lgi = run_comparison()


    def smooth(scalars, weight=0.95):
        if not scalars: return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed


    l_prox_smooth = smooth(l_prox)
    l_adamw_smooth = smooth(l_adamw)
    l_sadam_smooth = smooth(l_sadam)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='black')
    ax1.plot(l_prox_smooth, color='orange', alpha=0.7, label='Prox-SGD (Baseline)')
    ax1.plot(l_adamw_smooth, color='purple', alpha=0.7, label='AdamW (Baseline)')
    ax1.plot(l_sadam_smooth, color='blue', linewidth=2, label='S-Adam (Aggressive)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('LGI Score', color='green')
    lgi_smooth = smooth(lgi, 0.9)
    if lgi_smooth:
        ax2.fill_between(range(len(lgi_smooth)), lgi_smooth, color='green', alpha=0.15)

    plt.title("S-Adam vs Prox-SGD vs AdamW (TinyImageNet)")
    plt.tight_layout()
    plt.savefig('sadam_vs_proxsgd_adamw_tinyimagenet1.png', dpi=300)
    print("Saved plot to sadam_vs_proxsgd_adamw_tinyimagenet1.png")
    
    # --- Chattering Trajectory Visualization ---
    # [MODIFIED] Use synthetic Beale's function for this plot
    print("\nGenerating Synthetic Chattering Trajectories...")
    steps = 2000  # More steps for Beale's
    lr_syn = 0.01 # Lower LR for stability
    t_prox = run_synthetic_optimizer(ProxSGD, steps=steps, lr=lr_syn, momentum=0.9, l1_lambda=0.0)
    t_adamw = run_synthetic_optimizer(AdamW, steps=steps, lr=lr_syn)
    t_sadam = run_synthetic_optimizer(SAdam, steps=steps, lr=lr_syn, sigma=0.01, lgi_lambda=5.0, k_directions=5)

    plt.figure(figsize=(10, 8))
    
    # Plot Background Contours
    x_range = np.linspace(-2.5, 2.5, 200)
    y_range = np.linspace(-1.0, 4.0, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 10 * np.abs(Y - X**2) + np.abs(1 - X)
    plt.contourf(X, Y, np.log(Z + 1), levels=30, cmap='gray_r', alpha=0.6)
    
    # Convert list of tuples to numpy arrays for easier slicing
    t_prox = np.array(t_prox)
    t_adamw = np.array(t_adamw)
    t_sadam = np.array(t_sadam)

    # Plot trajectories
    plt.plot(t_prox[:, 0], t_prox[:, 1], color='orange', alpha=0.6, label='Prox-SGD', linewidth=1, marker='o', markersize=2)
    plt.plot(t_adamw[:, 0], t_adamw[:, 1], color='purple', alpha=0.6, label='AdamW', linewidth=1, marker='^', markersize=2)
    plt.plot(t_sadam[:, 0], t_sadam[:, 1], color='blue', alpha=0.8, label='S-Adam', linewidth=2, marker='*', markersize=3)
    
    # Mark start and end
    plt.scatter(t_prox[0, 0], t_prox[0, 1], marker='o', color='green', s=100, label='Start')
    plt.scatter(t_prox[-1, 0], t_prox[-1, 1], marker='x', color='orange', s=100, label='Prox-SGD End')
    plt.scatter(t_adamw[-1, 0], t_adamw[-1, 1], marker='x', color='purple', s=100, label='AdamW End')
    plt.scatter(t_sadam[-1, 0], t_sadam[-1, 1], marker='*', color='blue', s=200, label='S-Adam End')
    
    # Mark True Minimum
    plt.scatter(1.0, 1.0, marker='P', color='red', s=150, label='Global Min (1, 1)', zorder=10)

    plt.title("Optimization Trajectory on Nonsmooth Rosenbrock")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-1.0, 4.0)
    plt.tight_layout()
    plt.savefig('chattering_trajectory_tinyimagenet.png', dpi=300)
    print("Saved chattering plot to chattering_trajectory_tinyimagenet.png")

    plt.show()
