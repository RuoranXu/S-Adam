import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import zipfile
import tarfile
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
# ImageWoof Dataset
# ==============================================================================
class ImageWoof(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.join(root, 'imagewoof2-160')
        self.train = train
        self.transform = transform

        if download:
            self._download(root)

        if not os.path.exists(self.root):
            raise RuntimeError(f'Dataset root not found at {self.root}. You can use download=True to download it')

        self.data = []
        self.targets = []
        
        # Structure: root/train/class/xxx.jpg and root/val/class/xxx.jpg
        split = 'train' if train else 'val'
        split_dir = os.path.join(self.root, split)
        
        print(f"Looking for dataset at: {split_dir}")
        
        if not os.path.exists(split_dir):
            print(f"WARNING: Split directory {split_dir} does not exist!")
            # Fallback check: maybe the folder is just 'imagewoof2' or similar?
            # Or maybe no split folders?
            self.classes = []
        else:
            # Get classes (sorted for consistency)
            self.classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            cls_idx = self.class_to_idx[cls]
            valid_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in valid_files:
                self.data.append(os.path.join(cls_dir, img_name))
                self.targets.append(cls_idx)

        print(f"Total images loaded for {split}: {len(self.data)}")
        if len(self.data) == 0:
            print("CRITICAL ERROR: No images found. If the dataset exists but is empty, delete the folder and re-run.")

    def _download(self, root):
        if os.path.exists(os.path.join(root, 'imagewoof2-160')):
            print("Dataset already exists.")
            return

        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
        filename = "imagewoof2-160.tgz"
        filepath = os.path.join(root, filename)

        if not os.path.exists(filepath):
            print(f"Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)

        print("Extracting dataset...")
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(root)
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
# Part 3: Model Definition (ResNet18)
# ==============================================================================
def get_resnet_model(num_classes=10):
    # Using ResNet18
    model = models.resnet18(weights=None)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


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


def run_comparison():
    # Setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading ImageWoof...")
    dataset = ImageWoof('./data', train=True, download=True, transform=transform)
    test_dataset = ImageWoof('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 7

    # --- Run Prox-SGD (Baseline) ---
    print("\nTraining with Prox-SGD (Baseline)...")
    torch.manual_seed(42)
    model_prox = get_resnet_model(num_classes=10).to(device)
    opt_prox = ProxSGD(model_prox.parameters(), lr=0.01, momentum=0.9, l1_lambda=1e-4)

    losses_prox = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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

    max_mem_prox = 0
    if torch.cuda.is_available():
        max_mem_prox = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # --- Run AdamW (Standard Baseline) ---
    print("\nTraining with AdamW...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    model_adamw = get_resnet_model(num_classes=10).to(device)
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

    max_mem_adamw = 0
    if torch.cuda.is_available():
        max_mem_adamw = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # --- Run S-Adam (Ours) ---
    print("\nTraining with S-Adam (Ours)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    model_sadam = get_resnet_model(num_classes=10).to(device)

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

    max_mem_sadam = 0
    if torch.cuda.is_available():
        max_mem_sadam = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # === FINAL PRINT RESULT ===
    print("\n" + "=" * 60)
    print("      FINAL PERFORMANCE SUMMARY      ")
    print("=" * 60)
    print(f"{'Method':<20} | {'Total Time (s)':<15} | {'Max Memory (MB)':<15}")
    print("-" * 60)
    print(f"{'Prox-SGD (Base)':<20} | {total_time_prox:.4f}          | {max_mem_prox:.2f}")
    print(f"{'AdamW (Base)':<20} | {total_time_adamw:.4f}          | {max_mem_adamw:.2f}")
    print(f"{'S-Adam (Ours)':<20} | {total_time_sadam:.4f}          | {max_mem_sadam:.2f}")
    print("-" * 60)

    # Calculate Fairness Factor
    ratio = total_time_sadam / total_time_prox
    print(f"Cost Factor: S-Adam is {ratio:.2f}x slower than Prox-SGD.")
    print("=" * 40 + "\n")

    return losses_prox, losses_adamw, losses_sadam, opt_sadam.lgi_history


# ==============================================================================
# Visualization
# ==============================================================================
if __name__ == "__main__":
    sys.stdout = Logger('training_log_prox_adamw_sadam_imagewoof_resnet.txt')

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

    plt.title("S-Adam vs Prox-SGD vs AdamW (ImageWoof + ResNet)")
    plt.tight_layout()
    plt.savefig('sadam_vs_proxsgd_adamw_imagewoof_resnet.png', dpi=300)
    print("Saved plot to sadam_vs_proxsgd_adamw_imagewoof_resnet.png")
    plt.show()