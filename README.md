#  High-Performance Cyber Threat Detection using Graph Neural Networks 

A **GPU-accelerated and HPC-ready machine learning system** for detecting and classifying cyber threats using the **MachineLearningCVE** dataset (derived from CICIDS2017).  
The system leverages **Graph Neural Networks (GCNs)** and **MPI-based distributed training** to enable scalable, high-performance intrusion detection on large network datasets.

---

##  Problem Statement

###  Cybersecurity Challenges

Modern networks face a variety of cyberattacks such as **DDoS**, **DoS**, and **Port Scans**. Traditional Intrusion Detection Systems (IDS) face several bottlenecks:

####  Data Processing Limitations
- Millions of network flows with dozens of features make CPU-based preprocessing slow.  
- Real-time detection is difficult without GPU acceleration.

####  Attack Detection Gaps
- Signature-based IDS cannot detect **zero-day attacks**.  
- Classical ML models fail to capture inter-flow relationships.  
- Severe **class imbalance** (many benign flows vs few attacks) biases predictions.

####  Computational Bottlenecks
- Long training times for large datasets.  
- Multi-GPU scaling is non-trivial.  
- Graph-based deep learning often lacks HPC integration.

---

##  Solution Approach (HPC Ready)

This project implements a **distributed, GPU-accelerated pipeline** for cyber threat detection:

1. **GPU Preprocessing:** Uses **RAPIDS (cuDF, cuML, Dask-cuDF)** for high-speed data loading and cleaning.  
2. **Graph Representation:** Transforms network flows into **graphs**:  
   - **Nodes:** individual traffic flows  
   - **Edges:** similarity relationships via kNN  
   - **Labels:** traffic classes (Benign, DoS, DDoS, PortScan, Other)  
3. **Distributed GCN Training:**  
   - **PyTorch Geometric** for GCN modeling  
   - **MPI (via mpi4py)** for gradient synchronization  
   - Scales to multiple GPUs or CPU cores in HPC clusters  
4. **Evaluation:** Classification metrics, confusion matrices, and learning curve visualization.

---

##  Core Objectives

- Efficiently load and preprocess large CSV datasets on GPUs.  
- Balance dataset with undersampling and class weights.  
-  Build a kNN graph for relational modeling of network flows.  
-  Train a **Graph Convolutional Network (GCN)** on large graphs.  
-  Enable **distributed training on HPC clusters** with MPI.  
-  Evaluate using precision, recall, F1-score, confusion matrices, and runtime benchmarks.

---

##  Technical Innovation

| Feature | Innovation |
|---------|------------|
| End-to-End GPU Pipeline | From CSV ingestion → preprocessing → graph → distributed GCN training |
| Graph-Based Modeling | Captures inter-flow relationships in network traffic |
| Balanced Training | Weighted loss + undersampling handle class imbalance |
| Distributed HPC Training | Gradient averaging using MPI (AllReduce) ensures scalability |
| Multi-GPU Acceleration | Fully leverages CUDA devices for large datasets |

---

##  Technical Architecture

### Cyber Threat Detection Pipeline

    MachineLearningCVE CSVs
    ↓
    GPU-Accelerated Data Preprocessing (Dask-cuDF)
    ↓
    Unified cuDF DataFrame → Label Mapping & Class Grouping
    ↓
    Class-Balanced Undersampling
    ↓
    Feature Normalization (cuML StandardScaler)
    ↓
    kNN Graph Construction (5 neighbors)
    ↓
    PyTorch Geometric GCN Model
    ↓
    Distributed Training via MPI / DDP
    ↓
    Threat Classification
    ↓
    Evaluation: Classification Report & Confusion Matrix

---

##  Project Implementation Plan

### **Phase 1: Data Engineering & Preprocessing (2 weeks)**
- Load CSVs in parallel using **Dask-cuDF**.
- Merge into a single **cuDF DataFrame**.
- Clean column names and standardize labels.
- Map attack types:
  - DoS: GoldenEye, Hulk, Slowloris, Slowhttptest  
  - DDoS → DDoS  
  - PortScan → PortScan  
  - BENIGN → BENIGN  
  - Others → Other  
- Apply **undersampling** (max 50,000 samples/class).
- Retain **numeric features**, handle NaN/infinite values.
- Apply **StandardScaler** for normalization.

---

### **Phase 2: Graph Construction (1 week)**
- Use **cuML NearestNeighbors** for kNN graph generation.
- Create edge indices (`edge_index`) and symmetrize edges.
- Remove self-loops.
- Encode labels using **cuML LabelEncoder**.
- Perform stratified 80/20 train-test split.
- Save graph tensors: `x`, `edge_index`, `y`, `train_mask`, `test_mask`.

---

### **Phase 3: GNN Model Training (2 weeks)**
- **Model Architecture (GCN):**
- **Training Configuration:**
- Loss: Weighted NLLLoss
- Optimizer: Adam (lr=1e-2)
- Epochs: up to 200 (early stopping, patience=10)
- **Multi-GPU Training (DDP):**
- Each GPU processes a shard of training data.
- Metrics reduced using NCCL backend.

---

### **Phase 4: Evaluation & Reporting (1 week)**
- Compute:
- Accuracy
- Precision / Recall / F1-score
- Generate:
- Classification report
- Confusion matrix
- Compare single-GPU vs multi-GPU runtime.
- Analyze scalability and GPU utilization.

---

### **Phase 5: Deployment & Extension (Optional, 1–2 weeks)**
- Save trained GCN model (`gcn_model.pt`).
- Package for **real-time IDS integration**.
- Build a dashboard with:
- Attack distribution
- Real-time predictions
- Confusion matrix visualization

---

## ⚙️ Technical Specifications

| Component | Description |
|------------|-------------|
| **Dataset** | MachineLearningCVE (CICIDS2017) |
| **Language** | Python 3.9+ |
| **Platform** | windows with CUDA GPUs |
| **Core Libraries** |  PyTorch, PyTorch Geometric, scikit-learn |
| **HPC Setup** | PyTorch DDP with NCCL backend |
| **Outputs** | `features.npy`, `labels.npy`, `edge_index.npy`, `hpc_gcn_model.pth`, evaluation reports |

---
