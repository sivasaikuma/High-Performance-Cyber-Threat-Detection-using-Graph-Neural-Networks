# High-Performance-Cyber-Threat-Detection-using-Graph-Neural-Networks
A machine learning-based system for detecting and classifying cyber threats using the MachineLearningCVE dataset
## Problem Statement

### **Cybersecurity Challenges**

With the explosive growth of internet-connected devices and services, networks face a wide range of cyberattacks such as **DDoS, Port Scans, and DoS variants**. Traditional intrusion detection systems (IDS) struggle to keep pace due to:

**Data Processing Limitations**
- Network traffic logs contain millions of flows with dozens of features, making them computationally expensive to process.
- CPU-based preprocessing pipelines are too slow for real-time detection.

**Attack Detection Gaps**
- Rule-based IDS cannot detect **zero-day attacks**.
- Classical ML models (Decision Trees, SVM, etc.) treat flows independently and fail to capture **relationships between traffic patterns**.
- Severe **class imbalance** (huge number of benign flows vs fewer rare attacks) biases models.

**Computational Bottlenecks**
- Long training times for large datasets.
- Difficulty in scaling across multiple GPUs.
- Limited integration of GPU acceleration with graph-based deep learning.

---

## Solution Approach

This project builds a **GPU-accelerated cyber threat detection pipeline** using the **MachineLearningCVE dataset** (derived from CICIDS2017). The pipeline:

- Uses **RAPIDS (cuDF, cuML, Dask-cuDF)** for high-speed preprocessing directly on GPUs.
- Converts network flows into a **graph structure** where:
  - **Nodes** = traffic flows
  - **Edges** = similarity relationships (via kNN)
  - **Labels** = traffic class (Benign, DoS, DDoS, PortScan, Other)
- Trains a **Graph Convolutional Network (GCN)** using **PyTorch Geometric (PyG)** for classification.
- Scales across GPUs with **Distributed Data Parallel (DDP)**.
- Evaluates with **classification reports and confusion matrices**.

---

## Project Overview

### **Core Objectives**
1. **Efficient Data Handling**: Load and merge large CSVs on GPU.
2. **Balanced Dataset**: Apply undersampling to prevent class dominance.
3. **Graph Construction**: Build kNN similarity graphs for relational modeling.
4. **GNN Model Training**: Train GCN for accurate multi-class classification.
5. **Multi-GPU Scalability**: Use HPC with DDP for parallelism.
6. **Evaluation**: Provide classification report, confusion matrix, and runtime analysis.

### **Technical Innovation**
- **End-to-End GPU Pipeline**: From CSV parsing → scaling → graph → training.
- **Graph Representation of Traffic**: Captures contextual similarities among flows.
- **Balanced Training**: Class weights + undersampling to handle imbalance.
- **Scalable HPC Training**: DDP ensures linear scaling across multiple GPUs.

---

## Technical Architecture

### **Cyber Threat Detection Pipeline**
```
MachineLearningCVE CSV Files →
GPU-Accelerated Ingestion (Dask-cuDF) →
Merge into Unified cuDF DataFrame →
Label Mapping & Class Grouping →
Class-Balanced Undersampling →
Feature Scaling (cuML StandardScaler) →
kNN Graph Construction →
PyTorch Geometric GCN Model →
Training (Single-GPU / Multi-GPU DDP) →
Threat Classification →
Evaluation (Report & Confusion Matrix)
```

---

## Project Implementation Plan

### **Phase 1: Data Engineering & Preprocessing (2 weeks)**
- Load **all CSV files** in parallel with Dask-cuDF.
- Merge partitions into one GPU DataFrame (`cuDF`).
- Clean column names and labels.
- Map raw attack labels into grouped categories:
  - DoS GoldenEye, Hulk, Slowloris, Slowhttptest → **DoS**
  - DDoS → **DDoS**
  - PortScan → **PortScan**
  - BENIGN → **BENIGN**
  - Others → **Other**
- Perform **class-balanced undersampling** (cap at 50,000 records per class).
- Select numeric features only; handle missing/infinite values.
- Standardize features with **cuML StandardScaler**.

---

### **Phase 2: Graph Construction (1 week)**
- Build **kNN graph** on GPU using cuML NearestNeighbors.
- Construct directed edges (`edge_index`).
- Symmetrize edges and remove self-loops.
- Encode labels with **cuML LabelEncoder**.
- Perform stratified **train/test split (80/20)**.
- Save processed graph tensors (`x`, `edge_index`, `y`, train/test indices) for reuse.

---

### **Phase 3: GNN Model Training (2 weeks)**
- Define **Graph Convolutional Network (GCN)**:
  - Input → GCNConv → ReLU
  - Hidden → GCNConv → ReLU
  - Output → Linear → LogSoftmax
- Train using:
  - **Loss**: Weighted NLLLoss (class imbalance handling).
  - **Optimizer**: Adam, lr=1e-2.
  - **Epochs**: 200 max, with **early stopping (patience=10)**.
- **Single GPU**: Direct training loop.
- **Multi-GPU (DDP)**:
  - Each rank processes its shard of training data.
  - Metrics (loss, accuracy) reduced across GPUs using NCCL.

---

### **Phase 4: Evaluation & Reporting (1 week)**
- Compute metrics: Accuracy, Precision, Recall, F1-score.
- Generate **classification report** (per class performance).
- Produce **confusion matrix** to visualize misclassifications.
- Benchmark training time on **1 GPU vs multiple GPUs**.
- Analyze scalability and GPU utilization.

---

### **Phase 5: Deployment & Extension (Optional, 1–2 weeks)**
- Save final GCN model for reuse.
- Package pipeline for real-time IDS integration.
- Build a dashboard to visualize:
  - Attack type distribution
  - Real-time predictions
  - Confusion matrix results

---

## Technical Specifications
- **Dataset**: MachineLearningCVE (CICIDS2017)
- **Platform**: Linux / WSL2 with CUDA GPUs
- **Language**: Python 3.9+
- **Core Libraries**:
  - RAPIDS: cuDF, cuML, Dask-cuDF
  - PyTorch + PyTorch Geometric
  - scikit-learn for evaluation
- **HPC Setup**: PyTorch DistributedDataParallel with NCCL backend
- **Outputs**:
  - Cached graph (`graph_cache.pt`)
  - Trained GCN model
  - Classification report + confusion matrix
