# 🛡️ FraudShield AI — Graph Neural Network Financial Fraud Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.139.2-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.8.0-3C2179.svg?style=flat)](https://pyg.org)
[![React](https://img.shields.io/badge/React-19.2.7-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.1.1-646CFF.svg?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

**FraudShield AI** is an enterprise-grade financial fraud detection platform that combines **Graph Neural Networks (GraphSAGE / GraphConv with Residual Node Skip Connections)** with **LightGBM Gradient Boosted Decision Trees** to uncover complex, multi-hop transaction fraud rings and individual anomalous behaviors on the IEEE-CIS Fraud Detection dataset.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Benchmark Results](#-benchmark-results)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [1-Click Launch (Recommended)](#1-click-launch-recommended)
  - [Manual Installation & Setup](#manual-installation--setup)
- [Model Training & Pipeline Replication](#-model-training--pipeline-replication)
- [Explainability & XAI](#-explainability--xai)
- [API Reference](#-api-reference)
- [Testing & Integrity Auditing](#-testing--integrity-auditing)
- [Default Credentials](#-default-credentials)
- [License](#-license)

---

## 🔍 Overview & Architecture

Traditional fraud detection systems rely primarily on isolated tabular feature engineering (e.g. transaction amounts, timestamps, risk scores). However, modern financial fraud often operates via syndicated rings sharing cards, devices, digital footprints, and proxy networks across temporal windows.

FraudShield AI models the financial ecosystem as a **dynamic multi-relational transaction graph**:
1. **Nodes**: Individual financial transactions with 460+ cleaned tabular attributes (V-features, C-counts, D-time deltas, card specs).
2. **Edges**: Temporal k-nearest neighbor relationships linking transactions sharing critical attributes:
   - Card identifiers (`card1` to `card6`)
   - Device IDs (`DeviceInfo`, `DeviceType`)
   - Digital identities (`P_emaildomain`, `R_emaildomain`, `addr1`, `addr2`)
3. **Hybrid Residual GraphSAGE**:
   - **Graph Convolution Path**: Aggregates neighborhood signals with edge weights via `GraphConv`.
   - **Tabular Skip Pathway**: A 2-layer MLP residual connection ensuring isolated nodes retain full tabular predictive strength.
4. **Calibrated Ensemble**: Blends GNN neighborhood representations (0.2 weight) with LightGBM gradient boosting (0.8 weight) to maximize Precision-Recall AUC on heavily imbalanced real-world data (3.48% fraud rate).

---

## ✨ Key Features

- 🕸️ **Interactive 2D Force-Directed Graph Viewer**: Zoom, pan, and inspect transaction communities, hub nodes, and fraud clusters in real-time.
- ⚡ **Batch CSV Upload & Graph Inference Attachment**: Upload transaction batches and automatically link new transactions into the 590K-node historical graph via shared attributes.
- 🎯 **Dynamic Threshold & Operating Point Simulator**: Fine-tune precision vs. recall trade-offs live with real-time false-positive / false-negative cost calculations.
- 🔬 **Explainable AI (XAI)**: Local feature importance rankings, GNN neighbor contribution scores, and visual subgraph explanations.
- 🚨 **Analyst Alert Triage Queue**: Complete alert workflow management with dispositions (`Confirmed Fraud`, `Dismissed`, `Escalated`), persistent notes, and audit trails.
- 📊 **Executive Dashboard & Real-Time Analytics**: Live charts for transaction volume, risk score distributions, fraud rates across card networks, email domains, and operating systems.
- 📄 **Automated Audit & Compliance Reporting**: Generate and export executive PDF and Excel audit reports with detailed metrics and transaction logs.
- 🔐 **Secure Role-Based Access Control**: JWT authentication with Analyst and Admin access levels.

---

## 📈 Benchmark Results

Evaluated on the **IEEE-CIS Fraud Detection** dataset using a **strict chronological 70/15/15 temporal split** (88,581 test transactions, 3.48% positive rate):

| Model | Accuracy* | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flag Nothing (Baseline)** | 0.9652 | — | 0.0000 | 0.0000 | — | 0.0348 | — |
| **Logistic Regression** | 0.9389 | 0.2535 | 0.3886 | 0.3068 | 0.8249 | 0.1640 | 0.850 |
| **Random Forest** | 0.9694 | 0.5754 | 0.4655 | 0.5146 | 0.9057 | 0.5309 | 0.220 |
| **XGBoost** | 0.9707 | 0.6067 | 0.4483 | 0.5156 | 0.8923 | 0.5244 | 0.680 |
| **LightGBM** | 0.9710 | 0.6145 | 0.4509 | 0.5201 | 0.8872 | 0.5197 | 0.640 |
| **GraphSAGE (GNN)** | 0.9645 | 0.4883 | 0.4045 | 0.4424 | 0.8262 | 0.4147 | 0.590 |
| **🏆 Ensemble (GNN + LightGBM)** | **0.9724** | **0.6537** | **0.4428** | **0.5279** | **0.8926** | **0.5308** | **0.610** |

> **Note on Accuracy**: On severely imbalanced fraud datasets (3.5% fraud), a naive model that predicts "legitimate" for everything achieves 96.52% accuracy. Therefore, **PR-AUC and F1 Score** are the primary evaluation metrics.

---

## 🏛️ System Architecture

```
                                 ┌────────────────────────┐
                                 │ React 19 + Vite UI     │
                                 │ (Port 5173)            │
                                 └───────────┬────────────┘
                                             │ REST API / JWT
                                             ▼
                                 ┌────────────────────────┐
                                 │ FastAPI Backend Server │
                                 │ (Port 8000)            │
                                 └───────────┬────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
    │ Graph Neural Network │      │ LightGBM Classifier  │      │ SQLite DB & Reports  │
    │ (GraphSAGE + PyG)    │      │ (Tabular Booster)    │      │ (SQLAlchemy ORM)     │
    └──────────┬───────────┘      └──────────┬───────────┘      └──────────────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Soft-Voting Ensemble   │
                 │ 0.2 GNN + 0.8 LightGBM │
                 └────────────────────────┘
```

---

## 📂 Project Directory Structure

```
.
├── GNN.bat                         # 1-click all-in-one launcher
├── start_backend.bat               # Backend server launcher
├── start_frontend.bat              # Frontend dev server launcher
├── run_test.bat                    # Automated test runner
├── FIXES.md                        # Detailed engineering & audit history
├── requirements.txt                # Pinned Python dependencies
├── build_graph.py                  # Temporal graph builder & preprocessor
├── graph_sampler.py                # High-performance neighbor sampler
├── train.py                        # GraphSAGE model training script
├── train_ensemble.py               # LightGBM & Ensemble training script
├── evaluate.py                     # GraphSAGE evaluation & threshold tuning
├── baselines.py                    # ML baseline benchmark models
├── explainability.py               # GNNExplainer visualization script
├── verify_all.py                   # Complete independent test suite
├── validate_api.py                 # Live API endpoint validation
├── validate_artifacts.py           # Artifact consistency checker
│
├── models/
│   └── graphsage.py                # PyTorch Geometric GraphSAGE architecture
│
├── server/                         # FastAPI Backend
│   ├── main.py                     # Application factory & lifespan
│   ├── config.py                   # Paths, JWT, and model configurations
│   ├── database.py                 # SQLAlchemy models (User, AlertReview, Settings)
│   ├── api/                        # Route handlers (dashboard, predict, graph, etc.)
│   ├── auth/                       # JWT authentication & security router
│   └── ml/                         # ML singleton loader, pipeline & explainer
│
├── frontend/                       # React 19 + Vite Frontend
│   ├── package.json                # Node dependencies
│   ├── vite.config.js              # Vite server configuration
│   └── src/
│       ├── pages/                  # UI Pages (Dashboard, GraphViewer, AlertQueue, etc.)
│       ├── components/             # Reusable UI components & layouts
│       └── context/                # Authentication & application state
│
└── outputs/
    ├── models/                     # Serialized weights (best_model.pt, lightgbm.pkl, etc.)
    └── results/                    # Performance logs & benchmark metrics
```

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Operating System**: Windows 10/11 x64 (or Linux/macOS).
2. **Python**: Python 3.12.x installed and added to `PATH`.
3. **Node.js**: Node.js 18+ and `npm` installed.
4. **GPU (Optional)**: NVIDIA GPU with CUDA 12.6+ for accelerated training and graph inference.

---

### 1-Click Launch (Recommended)

Simply double-click **`GNN.bat`**.

The script automatically:
1. Verifies the Python virtual environment and model artifacts.
2. Runs `npm install` for frontend packages if missing.
3. Launches the FastAPI backend server on `http://127.0.0.1:8000`.
4. Waits for model initialization and prediction cache warm-up.
5. Launches the React Vite frontend on `http://127.0.0.1:5173`.
6. Opens the dashboard in your default browser.

---

### Manual Installation & Setup

#### 1. Setup Backend Python Environment

```powershell
# Create Python 3.12 virtual environment
py -3.12 -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install PyTorch with CUDA (or CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install required dependencies
pip install -r requirements.txt
```

#### 2. Start Backend Server

```powershell
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be accessible at: `http://127.0.0.1:8000/docs`

#### 3. Setup and Start Frontend

```powershell
cd frontend
npm install
npm run dev -- --port 5173 --host 127.0.0.1
```
Access the application at: `http://127.0.0.1:5173`

---

## 🔄 Model Training & Pipeline Replication

To reproduce the entire training pipeline from raw IEEE-CIS CSVs from scratch:

```powershell
# 1. Build the temporal multi-relational graph and preprocessor
python build_graph.py --split temporal

# 2. Generate edge attribute lookup index for real-time inference
python save_edge_attr_lookup.py

# 3. Cache raw node metadata for fast dashboard queries
python save_node_meta.py

# 4. Train the GraphSAGE neural network (~20 min on GPU)
python train.py

# 5. Train LightGBM tabular baseline & soft-voting ensemble
python train_ensemble.py

# 6. Evaluate baselines (Logistic Regression, Random Forest, XGBoost)
python baselines.py

# 7. Generate explainability plots and visualizations
python explainability.py
python visualize.py
```

---

## 🧠 Explainability & XAI

FraudShield AI provides multi-level model interpretability:
- **Global Feature Attribution**: Identified dominant predictive signals across 460+ columns (Transaction Amount, C1, C13, D1, D2, card1, V-features).
- **Local GNN Explanations**: Subgraph extraction isolating the exact neighbors, shared devices, and card connections contributing to an anomaly score.
- **Visual Subgraphs**: High-resolution neighborhood visualizations generated in `outputs/plots/`.

---

## 📡 API Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/auth/login` | Authenticate user and receive JWT bearer token | No |
| `GET` | `/health` | Service health status and loaded model confirmation | No |
| `GET` | `/api/dashboard` | High-level summary metrics, volume, and fraud stats | Yes |
| `GET` | `/api/transactions` | Paginated transaction list with risk scores & filters | Yes |
| `GET` | `/api/transactions/{id}` | Detailed transaction metadata and risk factors | Yes |
| `POST` | `/api/predict` | Real-time single transaction inference | Yes |
| `POST` | `/api/upload-predict` | Batch CSV upload with graph attachment scoring | Yes |
| `GET` | `/api/graph` | Subgraph neighborhood extraction for visualization | Yes |
| `GET` | `/api/explain/{id}` | GNNExplainer feature attributions for a node | Yes |
| `GET` | `/api/metrics` | Model performance, confusion matrix & ROC/PR curves | Yes |
| `GET` | `/api/alerts` | Alert review queue with analyst dispositions | Yes |
| `POST` | `/api/alerts/{id}/review` | Submit analyst review (`confirmed_fraud`/`dismissed`) | Yes |
| `GET` | `/api/reports/pdf` | Export executive PDF compliance report | Yes |
| `GET` | `/api/settings` | Retrieve active threshold and model selection | Yes |
| `POST` | `/api/settings` | Update fraud threshold override | Yes |

---

## 🧪 Testing & Integrity Auditing

The repository includes a suite of independent validation scripts:

```powershell
# 1. Validate artifact consistency and feature parity
python validate_artifacts.py

# 2. Validate live API endpoints against backend server
python validate_api.py

# 3. Test upload inference attachment on held-out samples
python verify_serving.py

# 4. Comprehensive independent end-to-end audit
python verify_all.py
```

---

## 🔑 Default Credentials

The platform comes pre-seeded with default demo credentials:

| Role | Username | Password | Permissions |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `fraud123` | Full administrative access, threshold tuning, and settings |
| **Fraud Analyst** | `analyst` | `demo456` | Alert triage, transaction analysis, and report generation |

---

## 📄 License

This project is licensed under the MIT License.
