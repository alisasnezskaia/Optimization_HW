# Gradient Methods for Semi-Supervised Learning

This project explores and compares optimization techniques for semi-supervised binary classification using kernel-based similarity. The primary focus is on:

- **Gradient Descent (GD)**
- **Block Coordinate Gradient Descent with Gauss-Southwell Rule (BCGD-GS)**
- **Coordinate Minimization (CM)**

---

## Problem Overview

We address binary classification with limited labeled data

### Datasets Used:
- **Synthetic dataset**: Two clusters, 10% labeled
- **Real-world dataset**: UCI Covertype (subset of two classes, 1,000 samples, 10% labeled)

---

## Methodology

### Similarity Measure
- **Gaussian RBF Kernel**:  
  \( K(x, y) = \exp(-\gamma \|x - y\|^2) \), with γ = 1.0

### Loss Function
Minimizes disagreement between similar samples:
- Between labeled and unlabeled points
- Among unlabeled points

### Optimization Algorithms

- **Gradient Descent (GD)**
  - Fixed step (1/L based on Lipschitz constant)
  - Armijo line search (adaptive step size)
  
- **Block Coordinate Gradient Descent (BCGD-GS)**
  - Updates one coordinate (unlabeled point) per iteration
  - Chooses coordinate via Gauss-Southwell rule

- **Coordinate Minimization (CM)**
  - Sequential coordinate-wise minimization
  - Follows Gauss-Seidel update scheme

---

## Results Summary

### Synthetic Dataset
- **CM** converged fastest
- **GD (Armijo)** was more stable and accurate than fixed-step GD
- **BCGD-GS** required most iterations

### Real Dataset (UCI Covertype)
- **GD (Armijo)** achieved best accuracy-time tradeoff
- **CM** was fastest but had occasional accuracy drops
- **BCGD-GS** was accurate but slowest

---

## Conclusions

- **Armijo GD** provides strong performance across datasets
- **Coordinate Minimization** is ideal for resource-constrained tasks
- **BCGD-GS** is accurate but computationally expensive
- **RBF kernel** effectively captures feature similarity in SSL setups

---

## Structure

```bash
.
├── data/                   # Synthetic + real dataset preprocessors
├── methods/                # GD, BCGD-GS, CM implementations
├── plots/                  # Loss/accuracy visualizations
├── utils/                  # Kernel functions, helpers
└── main.py                 # Entry point for experiments
