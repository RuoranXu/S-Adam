# S-Adam: Singularity-aware Adam

This repository contains experimental code for **Singularity-aware Adam (S-Adam)**, an optimizer for non-smooth deep learning objectives. S-Adam extends AdamW with a local geometric instability probe and an adaptive damping factor, aiming to reduce gradient chattering near non-smooth singularities such as ReLU kinks and quantization operators.

<a href='https://arxiv.org/abs/2605.29547'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/RuoranXu/S-Adam'><img src='https://img.shields.io/badge/Project-Page-green'></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
## Environment

The following environment is recommended for the experiments:

```text
Python==3.10.19
torch==2.9.1
torchvision==0.24.1
numpy==2.2.6
matplotlib==3.10.8
Pillow==12.1.0
psutil==7.2.1
```

Create and activate a virtual environment:

```bash
conda create -n sadam python=3.10.19
conda activate sadam
```

Install dependencies:

```bash
pip install torch==2.9.1 torchvision==0.24.1 numpy==2.2.6 matplotlib==3.10.8 Pillow==12.1.0 psutil==7.2.1
```

Experiments were run on NVIDIA A800 GPUs.


## Key Hyperparameters

Typical S-Adam settings used in the scripts:

```python
SAdam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    k_directions=2,
    sigma=0.01,
    lgi_lambda=2.0,
)
```

Important parameters:

- `k_directions`: number of randomized directional probes for LGI estimation.
- `sigma`: perturbation radius for randomized probing.
- `lgi_lambda`: damping intensity for high-LGI regions.
- `stabilize_eps`: numerical stabilizer in the LGI ratio.

The ablation scripts vary `k_directions` to evaluate the cost / accuracy trade-off.

## Using S-Adam in Your Own Training Loop

S-Adam requires a closure with a `backward` flag:

```python
optimizer = SAdam(model.parameters(), lr=1e-3, k_directions=2, sigma=0.01, lgi_lambda=2.0)

def closure(backward=True):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    if backward:
        loss.backward()
    return loss

loss = optimizer.step(closure)
```

The optimizer uses `closure(backward=True)` for the real gradient update and `closure(backward=False)` for no-gradient forward probes.

## Notes for Reproducibility

- The scripts are configured as standalone experiments rather than command-line tools.
- To change batch size, epochs, learning rate, or S-Adam hyperparameters, edit the constants inside the relevant script.
- ResNet experiments use torchvision ResNet18 with ImageNet weights when available.
- S-Adam performs additional forward passes per update, so runtime depends strongly on `k_directions`.
- CUDA memory is reported through `torch.cuda.max_memory_allocated()` when CUDA is available.

## Citation

If you use this code, please cite the associated paper. Add the final BibTeX entry here after publication.

```bibtex
@misc{xu2026singularityawareoptimizationrandomizedgeometric,
      title={Singularity-aware Optimization via Randomized Geometric Probing: Towards Stable Non-smooth Optimization}, 
      author={Ruoran Xu and Borong She and Xiaobo Jin and Qiufeng Wang},
      year={2026},
      eprint={2605.29547},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.29547}, 
}
```
