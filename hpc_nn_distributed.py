# hpc_gcn_distributed_full_updated.py
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import os
import random
from time import time
import matplotlib.pyplot as plt

# PyTorch Geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

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

# --------------------
# Define the GCN
# --------------------
class HPCGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super(HPCGCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.fc(x)

# --------------------
# Visualization Functions
# --------------------
def visualize_results(true_test, preds_test, train_losses, train_accs, class_names):
    # Confusion Matrix
    cm = confusion_matrix(true_test, preds_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    # Training Curves
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()

    plt.show()

# --------------------
# Training function
# --------------------
def train_model(data, args, rank, size, comm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"[INFO] Using device: {device}")

    X = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    # Split train/test indices (stratified)
    idx = np.arange(y.shape[0])
    train_idx, test_idx, y_train, y_test = train_test_split(
        idx, y.cpu().numpy(), test_size=0.2, random_state=42, stratify=y.cpu().numpy()
    )

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    test_idx  = torch.tensor(test_idx, dtype=torch.long, device=device)

    in_dim  = X.shape[1]
    out_dim = int(y.max().item() + 1)

    model = HPCGCN(in_dim, args.hid, out_dim, dropout=args.dropout).to(device)

    # ---------------- Weighted Loss for Imbalanced Classes ----------------
    class_counts = Counter(y[train_idx].cpu().numpy())
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(out_dim)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Partition training nodes across MPI ranks
    n_train = train_idx.shape[0]
    chunk_size = n_train // size
    start = rank * chunk_size
    end = n_train if rank == size - 1 else (rank + 1) * chunk_size
    local_train_idx = train_idx[start:end]

    if rank == 0:
        print(f"[INFO] Train nodes: {n_train} | per-rank: {len(local_train_idx)}")
        print(f"[INFO] Test nodes:  {len(test_idx)} | classes: {out_dim}")

    # Track training curves
    train_losses = []
    train_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(X, edge_index)
        loss_local = criterion(out[local_train_idx], y[local_train_idx])
        loss_local.backward()

        # -------- Gradient averaging across ranks --------
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_cpu = param.grad.detach().cpu().numpy().astype(np.float32, copy=True)
            comm.Allreduce(MPI.IN_PLACE, grad_cpu, op=MPI.SUM)
            grad_cpu /= float(size)
            param.grad.data.copy_(torch.from_numpy(grad_cpu).to(param.grad.device))
        # -------------------------------------------------

        optimizer.step()

        # Compute global loss (average across ranks)
        loss_global = np.array([loss_local.item()], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, loss_global, op=MPI.SUM)
        loss_global = float(loss_global[0] / size)

        # Eval on test set
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index)
            preds = logits[test_idx].argmax(dim=1)
            correct = (preds == y[test_idx]).sum().item()
            acc = correct / len(test_idx)

        # Save curves (only on rank 0)
        if rank == 0:
            train_losses.append(loss_global)
            train_accs.append(acc)
            if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
                print(f"[Epoch {epoch:04d}] loss={loss_global:.4f} | acc={acc:.4f}")

    # ---------------- Save model & visualize ----------------
    if rank == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/hpc_gcn_model.pth")
        print("[INFO] Model saved -> models/hpc_gcn_model.pth")

        # Classes (adjust according to your dataset)
        class_names = ['BENIGN', 'DDoS', 'DoS', 'PortScan', 'Other']
        print(f"\nClasses ({len(class_names)}): {class_names}")

        # Evaluate on test set
        preds_test = logits[test_idx].argmax(dim=1).cpu().numpy()
        true_test = y[test_idx].cpu().numpy()
        print("\n=== Test Set Classification Report ===")
        from sklearn.metrics import classification_report
        print(classification_report(true_test, preds_test, target_names=class_names, digits=4))

        # Visualize confusion matrix & training curves
        visualize_results(true_test, preds_test, train_losses, train_accs, class_names)

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)

    args, _ = parser.parse_known_args()  # ignore unknown args for Jupyter safety

    set_seed(args.seed)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("[INFO] Loading graph data...")

    # Load preprocessed artifacts
    features = np.load(os.path.join(args.data_dir, "features.npy"))
    labels   = np.load(os.path.join(args.data_dir, "labels.npy"))
    edge_idx = np.load(os.path.join(args.data_dir, "edge_index.npy"))

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edge_idx, dtype=torch.long)

    data = Data(x=X, y=y, edge_index=edge_index)

    train_model(data, args, rank, size, comm)

if __name__ == "__main__":
    main()
