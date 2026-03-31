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

from grok.measure import get_weights_fast
from phd.topology import calculate_ph_dim_gpu

TRAIN_SIZE = 5000
WEIGHTS_WINDOW_SIZE = 100

class KeyboardInterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        print("\nInterrupt received! Finishing current batch & saving model...")
        self.interrupted = True

class MNISTGrokker(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def forward(self, x):
        return self.layers(x)
    
    def fit(self, epochs=1e5, min_val_accuracy=0.98, load_path=None):
        self.to(self.device)
        print("DEVICE:", self.device)

        tensor_data = TensorDataset(self.all_inputs, self.all_labels)
        g = torch.Generator(device=self.device).manual_seed(42)
        val_size = len(self.all_inputs) - TRAIN_SIZE
        
        train_set, val_set = random_split(
            tensor_data,
            [TRAIN_SIZE, val_size],
            generator=g
        )

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=val_size, shuffle=False)

        filename = f"mnist_grok_{time.time()}.csv"
        headers = ["epoch", "step", "train_accuracy", "train_loss",
                   "val_accuracy", "val_loss", "phdim_0"]
        
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

        for epoch in range(start_epoch, epochs):
            self.train()

            total_train_loss = 0.0
            total_train_acc = 0.0

            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()

                coeff = inputs.size(0) / TRAIN_SIZE

                total_train_loss += coeff * loss.item()
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(labels, dim=1)
                    acc = (preds == targets).float().mean()
                    total_train_acc += coeff * acc.item()
                step += 1
            
            self.eval()

            total_val_loss = 0.0
            total_val_acc = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels.float())

                    coeff = inputs.size(0) / val_size

                    total_val_loss += coeff * loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(labels, dim=1)
                    acc = (preds == targets).float().mean()
                    total_val_acc += coeff * acc.item()
                    step += 1
            
            weights_window.append(get_weights_fast(self))
            calculate_phd = len(weights_window) == WEIGHTS_WINDOW_SIZE

            if epoch == self.next_epoch_to_log:
                self.next_epoch_to_log = max(
                    int(1.02 * self.next_epoch_to_log), self.next_epoch_to_log + 1
                )
                phdim_0 = calculate_ph_dim_gpu(
                        torch.stack(weights_window),
                        min_points=WEIGHTS_WINDOW_SIZE//10,
                        max_points=WEIGHTS_WINDOW_SIZE,
                        point_jump=WEIGHTS_WINDOW_SIZE//10
                    ) if calculate_phd else None
                
                row = [
                    epoch,
                    step,
                    total_train_acc,
                    total_train_loss,
                    total_val_acc,
                    total_val_loss,
                    phdim_0
                ]

                with open(filename, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            
            if calculate_phd:
                weights_window.pop(0)

            if total_val_acc >= min_val_accuracy:
                checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'weights_window': weights_window
                }
                torch.save(checkpoint, f"mnist-grok/epoch={epoch}-step={step}.ckpt")
                print(f"REACHED VALIDATION ACCURACY {min_val_accuracy}. ENDING TRAINING")
                break

            if stopper.interrupted:
                checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'weights_window': weights_window
                }
                torch.save(checkpoint, f"mnist-grok/epoch={epoch}-step={step}.ckpt")