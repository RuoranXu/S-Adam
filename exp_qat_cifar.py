import torch
import torch.nn as nn
import torch.nn.functional as F
from s_adam import SAdam
import torchvision
import torchvision.transforms as transforms

# 1. Fake Quantization Module (The Nonsmooth Source)
class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 4-bit quantization: values to integers 0-15
        scale = 16.0
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        # Gradient is passed through as 1, but the function was nonsmooth
        # This mismatch causes chattering in standard Adam
        return grad_output

class QuantizedConv2d(nn.Conv2d):
    def forward(self, input):
        w_quant = FakeQuantizeSTE.apply(torch.tanh(self.weight)) # Tanh to keep in range
        return F.conv2d(input, w_quant, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# 2. Simple QNN Model
class SimpleQNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Quantized Convs
        self.conv1 = QuantizedConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = QuantizedConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3. Training Loop
def run_qat_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CIFAR-10 (Subset for demo speed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Use only 1000 samples for quick demo
    indices = torch.arange(1000)
    trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleQNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Compare optimizers
    optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=0.005),
        'S-Adam': SAdam(model.parameters(), model_ref=model, lr=0.005, lambda_lgi=2.0)
    }

    results = {}

    for name, opt in optimizers.items():
        print(f"Running {name}...")
        model = SimpleQNN().to(device) # Reset model
        if name == 'S-Adam':
             opt = SAdam(model.parameters(), model_ref=model, lr=0.005, lambda_lgi=1.0)
        else:
             opt = torch.optim.Adam(model.parameters(), lr=0.005)

        loss_hist = []
        for epoch in range(10): # Short epoch for demo
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                def closure():
                    opt.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                if name == 'S-Adam':
                    # Functional call needs inputs to calculate JVP
                    loss = opt.step(closure, inputs=inputs, targets=labels, criterion=criterion)
                else:
                    loss = opt.step(closure)
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(trainloader)
            loss_hist.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        results[name] = loss_hist

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    for name, hist in results.items():
        plt.plot(hist, label=name)
    plt.title('QAT Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('qat_convergence.png')
    print("QAT Experiment Finished.")

if __name__ == '__main__':
    run_qat_experiment()
