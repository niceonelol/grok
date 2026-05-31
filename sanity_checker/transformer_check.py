import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# Local modules required for your environment
import grok
from phd import calculate_ph_dim 

# ==========================================
# 1. Dataset Generation
# ==========================================
def create_modular_addition_dataloaders(p, alpha, seed, train_batch_size=512):
    """Generates all p^2 pairs and splits them by fraction alpha."""
    a = torch.arange(p)
    b = torch.arange(p)
    A, B = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([A.flatten(), B.flatten()], dim=1)
    targets = (inputs[:, 0] + inputs[:, 1]) % p
    
    # Randomly subset fraction alpha for training
    train_idx, test_idx = train_test_split(
        range(len(inputs)), train_size=alpha, shuffle=True, random_state=seed
    )
    
    train_ds = TensorDataset(inputs[train_idx], targets[train_idx])
    test_ds = TensorDataset(inputs[test_idx], targets[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    # Test loader provides the entire test set in a single batch
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 2. Transformer Architecture
# ==========================================
class GrokkingTransformer(nn.Module):
    def __init__(self, p, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Embedding(2, d_model) # 2 tokens: a and b
        
        # Pre-layer-norm standard encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            dropout=0.0, 
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.ln_final = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, p)

    def forward(self, x):
        # x shape: (Batch, 2)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Sum token and learned positional embeddings
        x = self.tok_emb(x) + self.pos_emb(positions)
        
        # Pass through 2-layer encoder
        encoded = self.encoder(x)
        
        # Extract hidden state at the SECOND token position (index 1)
        hidden = encoded[:, 1, :]
        
        # Final layer norm and linear projection
        logits = self.readout(self.ln_final(hidden))
        return logits

# ==========================================
# 3. Main Training Script
# ==========================================
def get_infinite_batches(dataloader):
    """Yields batches infinitely to train by steps rather than epochs."""
    while True:
        for batch in dataloader:
            yield batch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    primes = [113, 149, 197]
    alphas = [0.20, 0.25, 0.30]
    seeds = list(range(46, 51))
    total_steps = 60000
    eval_interval = 500
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize CSV
        
    for p in primes:
        for alpha in alphas:
            for seed in seeds:
                with open(f'sanity_checker/transformer_data/{p}_{alpha}_{seed}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Step', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc', 'phdim_0'])

                    torch.manual_seed(seed)
                    
                    train_loader, test_loader = create_modular_addition_dataloaders(p, alpha, seed)
                    batch_iterator = iter(get_infinite_batches(train_loader))
                    
                    model = GrokkingTransformer(p=p).to(device)
                    
                    optimizer = AdamW(
                        model.parameters(), 
                        lr=3e-3, 
                        betas=(0.9, 0.98), 
                        eps=1e-6, 
                        weight_decay=0.1
                    )
                    
                    # Brief linear warm-up over first 10 steps
                    scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)
                    
                    model.train()
                    for step in range(1, total_steps + 1):
                        x_batch, y_batch = next(batch_iterator)
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        
                        optimizer.zero_grad()
                        logits = model(x_batch)
                        loss = criterion(logits, y_batch)
                        loss.backward()
                        optimizer.step()
                        
                        if step <= 10:
                            scheduler.step()
                            
                        # Logging and Evaluation
                        if step % eval_interval == 0:
                            model.eval()
                            with torch.no_grad():
                                # Train metrics (on current batch)
                                train_loss = loss.item()
                                train_preds = torch.argmax(logits, dim=1)
                                train_acc = (train_preds == y_batch).float().mean().item()
                                
                                # Test metrics (on full test set)
                                x_test, y_test = next(iter(test_loader))
                                x_test, y_test = x_test.to(device), y_test.to(device)
                                
                                test_logits = model(x_test)
                                test_loss = criterion(test_logits, y_test).item()
                                test_preds = torch.argmax(test_logits, dim=1)
                                test_acc = (test_preds == y_test).float().mean().item()
                                
                                # Calculate PH dimension (H0) using the dummy function
                                ph_dim_h0 = calculate_ph_dim(model.tok_emb.weight.data)
                                
                                writer.writerow([step, 
                                    train_loss, train_acc, 
                                    test_loss, test_acc, 
                                    ph_dim_h0
                                ])
                                
                            model.train()

if __name__ == "__main__":
    main()