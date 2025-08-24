# hpc_nn_distributed.py
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import random
from time import time

# --------------------
# Utils
# --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)

# --------------------
# Define the neural network
# --------------------
class HPCNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super(HPCNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# --------------------
# Training function
# --------------------
def train_model(features, labels, args, rank, size, comm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"[INFO] Using device: {device}")

    # Normalize features (fit on full data for simplicity; ok when all ranks load the same files)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split dataset into train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Convert to tensors on device
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test,  dtype=torch.long, device=device)

    # Partition training data across MPI ranks (contiguous split)
    train_size = X_train.shape[0]
    chunk_size = train_size // size
    start = rank * chunk_size
    end = train_size if rank == size - 1 else (rank + 1) * chunk_size

    X_local = X_train[start:end]
    y_local = y_train[start:end]

    in_dim  = X_train.shape[1]
    out_dim = int(np.unique(labels).shape[0])

    # Class weights from global training labels (computed locally — acceptable because all ranks read same data)
    class_counts = Counter(y_train.detach().cpu().numpy())
    class_weights = np.zeros(out_dim, dtype=np.float32)
    for c in range(out_dim):
        class_weights[c] = 1.0 / max(class_counts.get(c, 1), 1)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = HPCNet(in_dim, args.hid, out_dim, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Simple minibatch loop per rank
    n_local = X_local.shape[0]
    batch_size = max(1, args.batch_size)
    n_batches = math.ceil(n_local / batch_size)

    if rank == 0:
        print(f"[INFO] Train samples: {train_size} | per-rank: {n_local} | batches/rank: {n_batches}")
        print(f"[INFO] Test samples:  {X_test.shape[0]} | classes: {out_dim}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_local = 0.0
        t0 = time()

        # shuffle indices locally each epoch
        perm = torch.randperm(n_local, device=device)
        X_local = X_local[perm]
        y_local = y_local[perm]

        for b in range(n_batches):
            s = b * batch_size
            e = min(n_local, s + batch_size)
            xb = X_local[s:e]
            yb = y_local[s:e]

            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()

            # -------- Gradient averaging across ranks (HPC fix) --------
            for param in model.parameters():
                if param.grad is None:
                    continue
                # Move grad to CPU numpy for MPI Allreduce
                grad_cpu = param.grad.detach().cpu().numpy().astype(np.float32, copy=True)
                comm.Allreduce(MPI.IN_PLACE, grad_cpu, op=MPI.SUM)
                grad_cpu /= float(size)
                # Write back to original grad tensor on its device
                param.grad.data.copy_(torch.from_numpy(grad_cpu).to(param.grad.device))
            # -----------------------------------------------------------

            optimizer.step()
            epoch_loss_local += loss.item() * (e - s)

        # Reduce epoch loss (sum) across ranks then normalize by global train size
        epoch_loss_global = np.array([epoch_loss_local], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, epoch_loss_global, op=MPI.SUM)
        epoch_loss_global = float(epoch_loss_global[0] / train_size)

        # ----- Global evaluation (accuracy) -----
        model.eval()
        with torch.no_grad():
            # local correctness on full test set (each rank computes on full test set)
            preds_local = model(X_test).argmax(dim=1)
            correct_local = (preds_local == y_test).sum().item()
            total_local = y_test.numel()

            # Average by summing once (only one copy per rank; sum then divide by size is fine,
            # but we want the global acc, so sum corrects and totals)
            correct_global = np.array([correct_local], dtype=np.int64)
            total_global   = np.array([total_local],   dtype=np.int64)
            comm.Allreduce(MPI.IN_PLACE, correct_global, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, total_global,   op=MPI.SUM)

            acc_global = float(correct_global[0]) / float(total_global[0])

        if rank == 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            dt = time() - t0
            print(f"[Epoch {epoch:04d}] loss={epoch_loss_global:.4f} | acc={acc_global:.4f} | time={dt:.2f}s")

    if rank == 0:
        print("Training finished successfully!")

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("[INFO] Loading data...")

    # Simple per-rank local load (shared FS). Switch to broadcast if needed.
    features = np.load(os.path.join(args.data_dir, "features.npy"))
    labels   = np.load(os.path.join(args.data_dir, "labels.npy"))

    train_model(features, labels, args, rank, size, comm)

if __name__ == "__main__":
    main()
