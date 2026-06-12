import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import csv
import signal
import time
import os
import random
import argparse

from grok.measure import get_weights_fast, E_alpha
from phd.topology import calculate_ph_dim_gpu

TRAIN_SIZE = 2500
WEIGHTS_WINDOW_SIZE = 500
PHDIM_WINDOW = 100

import random

def set_seed(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KeyboardInterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.interrupted = True

class MNISTGrokker(nn.Module):
    def __init__(self, scale_factor=4.0, train_size=2500):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_size = train_size
        raw_mnist = datasets.MNIST(root="mnist-grok/data", train=True, download=True)

        self.all_inputs = (raw_mnist.data.float() / 255.0).view(-1, 784).to(self.device)
        self.all_labels = F.one_hot(raw_mnist.targets, num_classes=10).float().to(self.device)
        
        self.inp_size, self.width, self.out_size = 784, 200, 10

        self.layers = nn.Sequential(
            nn.Linear(self.inp_size, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.out_size)
        )

        self.optimizer = optim.AdamW(self.parameters())
        self.criterion = nn.MSELoss()

        self.batch_size = 500

        self.next_epoch_to_log = 0
        self.next_epoch_to_print = 0

        self.scale_factor = scale_factor
        self.initialize_weights(self.scale_factor)
    
    def initialize_weights(self, scale_factor=4.0):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode='fan_in', nonlinearity='relu'
                )

                with torch.no_grad():
                    layer.weight.mul_(scale_factor)
                
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.layers(x)
    
    def fit(self, epochs=10000, min_val_accuracy=0.98, load_path=None, regularise='', eps=0.5, seed=46):
        assert regularise in ['', 'phd_L1', 'phd_L2']
        self.to(self.device)
        print("DEVICE:", self.device)

        tensor_data = TensorDataset(self.all_inputs, self.all_labels)
        g = torch.Generator(device="cpu").manual_seed(seed)
        val_size = len(self.all_inputs) - self.train_size
        
        train_set, val_set = random_split(
            tensor_data,
            [self.train_size, val_size],
            generator=g
        )

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=val_size, shuffle=False)

        filename = f"mnist-grok/mnist_grok_{self.train_size}_{self.scale_factor}_seed{seed}.csv"
        headers = ["epoch", "step", "train_accuracy", "train_loss",
                   "val_accuracy", "val_loss", "phdim_0", "e_alpha"]
        
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        if load_path is not None and os.path.exists(load_path):
            load = torch.load(load_path)
            self.load_state_dict(load['model_state_dict'])
            self.optimizer.load_state_dict(load['optimizer_state_dict'])
            step = load['step']
            start_epoch = load['epoch'] + 1
            weights_window = load['weights_window']
        else:
            step = 0
            weights_window = []
            start_epoch = 0

        stopper = KeyboardInterruptHandler()
        interrupt_flag = False

        for epoch in range(start_epoch, epochs):
            self.train()

            total_train_loss = 0.0
            total_train_acc = 0.0

            calculate_phd = len(weights_window) == WEIGHTS_WINDOW_SIZE
            phdim_0, e_alpha = None, None
            temp_weights = get_weights_fast(self)

            for inputs, labels in train_loader:

                self.optimizer.zero_grad()
                outputs = self(inputs)

                phd_loss = 0

                weights_window.append(get_weights_fast(self))
                calculate_phd = len(weights_window) == WEIGHTS_WINDOW_SIZE

                """
                if regularise:
                    if calculate_phd:
                        phdim_0 = calculate_ph_dim_gpu(
                            torch.stack(weights_window),
                            min_points=WEIGHTS_WINDOW_SIZE//10,
                            max_points=WEIGHTS_WINDOW_SIZE,
                            point_jump=WEIGHTS_WINDOW_SIZE//10
                        )
                        if regularise == 'phd_L1':
                            phd_loss = torch.abs(phdim_0 - 4)
                        else:
                            phd_loss = (phdim_0 - 4) ** 2
                """
                
                if calculate_phd:
                    if regularise:
                        phdim_0 = calculate_ph_dim_gpu(
                            torch.stack(weights_window),
                            min_points=WEIGHTS_WINDOW_SIZE//5,
                            max_points=WEIGHTS_WINDOW_SIZE,
                            point_jump=WEIGHTS_WINDOW_SIZE//50
                        )
                    if regularise == 'phd_L1':
                        phd_loss = torch.abs(phdim_0 - 4)
                    elif regularise == 'phd_L2':
                        phd_loss = (phdim_0 - 4) ** 2

                    temp_weights = weights_window.pop(0)

                loss = self.criterion(outputs, labels.float()) + eps * phd_loss
                loss.backward()
                self.optimizer.step()

                coeff = inputs.size(0) / self.train_size

                total_train_loss += coeff * loss.item()
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(labels, dim=1)
                    acc = (preds == targets).float().mean()
                    total_train_acc += coeff * acc.item()
                step += 1
            
            """
            if not regularise:
                weights_window.append(get_weights_fast(self))
                calculate_phd = len(weights_window) == WEIGHTS_WINDOW_SIZE
            """

            if epoch == self.next_epoch_to_log:
                self.next_epoch_to_log = max(
                    int(1.02 * self.next_epoch_to_log), self.next_epoch_to_log + 1
                )
                
                self.eval()

                total_val_loss = 0.0
                total_val_acc = 0.0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels.float()) + eps * phd_loss

                        coeff = inputs.size(0) / val_size

                        total_val_loss += coeff * loss.item()
                        preds = torch.argmax(outputs, dim=1)
                        targets = torch.argmax(labels, dim=1)
                        acc = (preds == targets).float().mean()
                        total_val_acc += coeff * acc.item()
                    
                    if not regularise and calculate_phd:
                        stacked_weights = torch.stack([temp_weights] + weights_window)
                        e_alpha = E_alpha(stacked_weights)
                        phdim_0 = calculate_ph_dim_gpu(
                            stacked_weights[-1*PHDIM_WINDOW:],
                            min_points=PHDIM_WINDOW//10,
                            max_points=PHDIM_WINDOW,
                            point_jump=PHDIM_WINDOW//10
                        )
                
                row = [
                    epoch,
                    step,
                    100 * total_train_acc,
                    total_train_loss,
                    100 * total_val_acc,
                    total_val_loss
                ]

                if phdim_0 is not None:
                  row.append(phdim_0.item())
                else:
                  row.append(None)

                if e_alpha is not None:
                    row.append(e_alpha)
                else:
                    row.append(None)

                with open(filename, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                if total_val_acc >= min_val_accuracy:
                    checkpoint = {
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }
                    torch.save(checkpoint, f"mnist-grok/epoch={epoch}-step={step}.ckpt")
                    print(f"REACHED VALIDATION ACCURACY {min_val_accuracy}. ENDING TRAINING")
                    break
            
            """
            if calculate_phd and not regularise:
                weights_window.pop(0)
            """

            if epoch == self.next_epoch_to_print:
                print(f"Epoch: {epoch}, Train Acc: {total_train_acc}, Val Acc: {total_val_acc}")
                self.next_epoch_to_print = max(self.next_epoch_to_print + 100,
                                                self.next_epoch_to_log)

            if stopper.interrupted:
                interrupt_flag = True
                print("\nInterrupt received! Finishing current batch & saving model...")
                checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                torch.save(checkpoint, f"mnist-grok/epoch={epoch}-step={step}_seed={seed}.ckpt")
                break

        if not interrupt_flag:
            print("Finished fitting model")
            checkpoint = {
            'epoch': epochs,
            'step': step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, f"mnist-grok/epoch={epochs}-step={step}-seed={seed}.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST Grokker")
    parser.add_argument("--train_size", type=int, default=2500, help="Size of the training dataset")
    parser.add_argument("--seed", type=int, default=46, help="Random seed for reproducibility")
    parser.add_argument("--scale_factor", type=float, default=5.0, help="Weight initialization scale factor")
    args = parser.parse_args()

    set_seed(args.seed)
    
    model = MNISTGrokker(scale_factor=args.scale_factor, train_size=args.train_size)
    model.fit(epochs=100000, eps=0.0, seed=args.seed)