import os
import csv
import math
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# This class was taken from Birdal et al. [6]
# ---------------------------------------------------------------------------
class AlexNet(nn.Module):
    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.size = self.get_size()
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def get_size(self):
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_full_network_weights_gpu(model):
    w_fragments = []
    for param in model.parameters():
        w_fragments.append(param.detach().view(-1))
    return torch.cat(w_fragments)

# Taken from Birdal et al. [6]
def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]


# Taken from Birdal et al. [6]
def calculate_ph_dim_gpu(W, min_points=200, max_points=1000, 
        point_jump=50, h_dim=0, print_error=False):
    from torchph.torchph.pershom import vr_persistence
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        samples = sample_W(W, n)
        dist_matrix = torch.cdist(samples, samples)
        
        d, _ = vr_persistence(dist_matrix, 0, 0)
        d = d[0]
        lengths.append((d[:, 1] - d[:, 0]).sum())

    lengths = torch.stack(lengths)
    
    # compute our ph dim by running a linear least squares
    x = torch.tensor(test_n).to(lengths).log()
    y = lengths.log()
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)

def get_dataloaders(batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'sanity_checker/phdim_data'
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = range(46, 51)
    batch_sizes = [64, 100, 128]
    target_iterations = 100000

    for seed in seeds:
        for batch_size in batch_sizes:
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            train_loader, test_loader = get_dataloaders(batch_size, seed)
            
            iters_per_epoch = len(train_loader)
            exact_epochs = target_iterations / iters_per_epoch
            total_epochs = math.ceil(exact_epochs / 100.0) * 100
            checkpoint_interval = total_epochs // 100

            print(f"Seed: {seed} | Batch: {batch_size} | Iters/Epoch: {iters_per_epoch} | Total Epochs: {total_epochs}")
            
            model = AlexNet(num_classes=10).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            trajectory = deque(maxlen=1000)
            csv_path = os.path.join(output_dir, f'{batch_size}_{seed}.csv')
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_accuracy', 'test_accuracy', 'phdim'])
                
                for epoch in range(1, total_epochs + 1):
                    model.train()
                    correct_train = 0
                    total_train = 0

                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        _, predicted = outputs.max(1)
                        total_train += targets.size(0)
                        correct_train += predicted.eq(targets).sum().item()
                        
                        trajectory.append(get_full_network_weights_gpu(model))
                    
                    if epoch % checkpoint_interval == 0:
                        train_acc = 100. * correct_train / total_train
                        
                        correct_test = 0
                        total_test = 0
                        model.eval() 
                        
                        with torch.no_grad(): 
                            for test_inputs, test_targets in test_loader:
                                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                                test_outputs = model(test_inputs)
                                _, test_predicted = test_outputs.max(1)
                                
                                total_test += test_targets.size(0)
                                correct_test += test_predicted.eq(test_targets).sum().item()
                                
                        test_acc = 100. * correct_test / total_test
                        
                        if len(trajectory) == 1000:
                            phdim = calculate_ph_dim_gpu(torch.stack(list(trajectory)), min_points=200, max_points=1000, point_jump=50)
                        else:
                            phdim = None
                        
                        print(f"Epoch: {epoch} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | PH Dim: {phdim}")
                            
                        writer.writerow([epoch, train_acc, test_acc, phdim])
                        f.flush()

if __name__ == "__main__":
    main()