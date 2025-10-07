import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "MachineLearningCVE"
OUTPUT_DIR = "artifacts"
MAX_SAMPLES_PER_CLASS = 40000  # desired per-class samples

def prepare_data(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR):
    # -----------------------------
    # 1. Load CSVs
    # -----------------------------
    print("[INFO] Loading CSV files...")
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(df)} rows from {len(files)} files")

    # Strip whitespace
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].str.strip()

    # -----------------------------
    # 2. Group labels
    # -----------------------------
    mapping = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
        'DoS GoldenEye': 'DoS',
        'DoS Hulk': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DoS slowloris': 'DoS'
    }
    df['Label_grp'] = df['Label'].map(mapping).fillna('Other')
    print("[INFO] Class counts (before undersampling):")
    print(df['Label_grp'].value_counts())

    # -----------------------------
    # 3. Report missing samples
    # -----------------------------
    print("\n[INFO] Samples needed to reach 40k per class:")
    for cls, count in df['Label_grp'].value_counts().items():
        needed = MAX_SAMPLES_PER_CLASS - count
        if needed > 0:
            print(f" - {cls}: need {needed} more samples")
        else:
            print(f" - {cls}: enough samples ({count})")

    # -----------------------------
    # 4. Undersample / limit to MAX_SAMPLES_PER_CLASS
    # -----------------------------
    dfs = []
    for cls, grp in df.groupby('Label_grp'):
        if len(grp) > MAX_SAMPLES_PER_CLASS:
            grp = grp.sample(MAX_SAMPLES_PER_CLASS, random_state=42)
        dfs.append(grp)
    df_bal = pd.concat(dfs, ignore_index=True)
    print("\n[INFO] After undersampling:")
    print(df_bal['Label_grp'].value_counts())
    print(f"[INFO] Total samples: {len(df_bal)}")

    # -----------------------------
    # 5. Features and labels
    # -----------------------------
    y = df_bal["Label_grp"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = df_bal.drop(columns=["Label", "Label_grp"])
    X = X.select_dtypes(include=[np.number])

    # -----------------------------
    # 6. Normalize
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # 7. kNN adjacency graph
    # -----------------------------
    print("[INFO] Building adjacency graph (kNN, k=5)...")
    adj = kneighbors_graph(X_scaled, n_neighbors=5, mode='connectivity', include_self=False)
    adj = coo_matrix(adj)
    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    edge_index = torch.stack([row, col], dim=0)

    # -----------------------------
    # 8. Save artifacts
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features.npy"), X_scaled)
    np.save(os.path.join(output_dir, "labels.npy"), y_encoded)
    np.save(os.path.join(output_dir, "edge_index.npy"), edge_index.numpy())
    print(f"[INFO] Saved features, labels, and edge_index -> {output_dir}")
    print(f"[INFO] Final: samples={len(df_bal)} | classes={len(np.unique(y_encoded))}")

if __name__ == "__main__":
    prepare_data()
