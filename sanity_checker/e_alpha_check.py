import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import deque
import numpy as np
import csv
import os
from scipy.spatial.distance import cdist

import phd

from gph.python import ripser_parallel
from ripser import ripser

def E_alpha(dist_matrix: np.ndarray, h_dim=0, **kwargs) -> float:

    diagrams = ripser_parallel(dist_matrix, maxdim=h_dim, n_threads=-1, metric="precomputed")['dgms']
    
    d = diagrams[h_dim]
    d = d[d[:, 1] < np.inf]

    alpha_sums = []

    for alpha in [0.5, 1.0, 2.0]:
        alpha_sum = np.power((d[:, 1] - d[:, 0]), alpha).sum()
        alpha_sums.append(alpha_sum)

    return alpha_sums

def get_full_network_weights(model):

    w_fragments = []
    for param in model.parameters():
        w_fragments.append(param.detach().cpu().view(-1))
    return torch.cat(w_fragments).numpy()

def get_full_network_weights_gpu(model):

    w_fragments = []
    for param in model.parameters():
        w_fragments.append(param.detach().view(-1))
    return torch.cat(w_fragments)

def create_modular_addition_dataloaders(p=97, alpha_split=0.4, seed=42, batch_size=512):
    
    a = torch.arange(p)
    b = torch.arange(p)
    A, B = torch.meshgrid(a, b, indexing='ij')
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    targets = (A_flat + B_flat) % p
    
    one_hot_A = F.one_hot(A_flat, num_classes=p).float()
    one_hot_B = F.one_hot(B_flat, num_classes=p).float()
    inputs = torch.cat([one_hot_A, one_hot_B], dim=-1)
    
    train_idx, test_idx = train_test_split(
        range(len(inputs)), 
        train_size=alpha_split, 
        shuffle=True, 
        random_state=seed
    )
    
    train_ds = TensorDataset(inputs[train_idx], targets[train_idx])
    test_ds = TensorDataset(inputs[test_idx], targets[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    
    return train_loader, test_loader

class GrokkingMLP(nn.Module):
    def __init__(self, p=97, hidden_features=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(p * 2, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, p)
        )

    def forward(self, x):
        return self.layers(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    p = 97
    train_split = 0.4
    epochs = 20000
    batch_size = 512
    seeds = list(range(46, 51))
    
    configs = [
        (0.15, 2e-4),
        (0.15, 3e-4),
        (0.25, 2e-3),
        (0.20, 2e-3)
    ]

    alphas = [0.5, 1.0, 2.0] 

    for seed in seeds:  
        for lr, wd in configs:
            with open(f'sanity_checker/e_alpha_data/{lr}_{wd}_{seed}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc', 'E_0.5', 'E_1.0', 'E_2.0'])

                print(f"\n--- Starting Run: LR={lr}, WD={wd}, Seed={seed} ---")

                torch.manual_seed(seed)
                
                train_loader, test_loader = create_modular_addition_dataloaders(
                    p=p, alpha_split=train_split, seed=seed, batch_size=batch_size
                )
                
                model = GrokkingMLP(p=p).to(device)
                
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                criterion = nn.CrossEntropyLoss()
                
                weights_window = deque(maxlen=5000)
                
                for epoch in range(1, epochs + 1):
                    model.train()
                    total_train_loss = 0
                    train_correct = 0
                    train_total = 0
                    
                    for x_batch, y_batch in train_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(x_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        
                        total_train_loss += loss.item() * x_batch.size(0)
                        preds = torch.argmax(outputs, dim=1)
                        train_correct += (preds == y_batch).sum().item()
                        train_total += x_batch.size(0)
                    
                        current_weights = get_full_network_weights_gpu(model)
                        weights_window.append(current_weights)
                    
                    e_alpha_bwd = None
                    e_alpha_fwd = None

                    e_half, e_1, e_2 = None, None, None
                    
                    if len(weights_window) == 5000 and (epoch % 200 == 0 or epoch == epochs):
                        bwd_trajectory_matrix = torch.stack(list(weights_window))
                        
                        bwd_dist_matrix = torch.cdist(bwd_trajectory_matrix, bwd_trajectory_matrix, p=2.0).cpu().numpy()
                        
                        e_half, e_1, e_2 = E_alpha(bwd_dist_matrix, h_dim=0)
                        
                        model.eval()
                        test_loss = 0
                        test_correct = 0
                        test_total = 0
                        
                        with torch.no_grad():
                            for x_test, y_test in test_loader:
                                x_test, y_test = x_test.to(device), y_test.to(device)
                                test_outputs = model(x_test)
                                test_loss += criterion(test_outputs, y_test).item() * x_test.size(0)
                                test_preds = torch.argmax(test_outputs, dim=1)
                                test_correct += (test_preds == y_test).sum().item()
                                test_total += x_test.size(0)
                                
                        train_acc = train_correct / train_total
                        avg_train_loss = total_train_loss / train_total
                        test_acc = test_correct / test_total
                        avg_test_loss = test_loss / test_total
                        
                        print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | E_alphas: {e_half:.4f}, {e_1:.4f}, {e_2:.4f}")
                        
                        writer.writerow([epoch,
                            avg_train_loss, train_acc,
                            avg_test_loss, test_acc,
                            e_half, e_1, e_2
                        ])

if __name__ == "__main__":
    main()