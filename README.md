# Capturing Long-Range Dependencies in Sequence Modeling: A Comparative Analysis of Recurrent and Self-Attention Architectures

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research_Complete-success)]()

## 📌 Project Overview

This research project conducts a rigorous empirical comparison between **Recurrent Neural Networks (RNNs)** and **Transformer (Self-Attention)** architectures. The primary objective is to evaluate their efficacy in capturing **long-range semantic dependencies** in Natural Language Processing (NLP) tasks.

While RNNs (including LSTMs and GRUs) have historically been the standard for sequence modeling, they suffer from the **sequential bottleneck** and difficulty in retaining long-term context due to vanishing gradients. This project implements a deep **Transformer Encoder from scratch** to demonstrate how **Self-Attention** mechanisms resolve these issues by reducing the path length between any two positions in a sequence to $O(1)$.

## 🚀 Key Contributions

- **Architectural Implementation**:
    - Implemented **Vanilla RNN**, **LSTM**, **GRU**, and **Bi-LSTM** models to establish strong baselines.
    - Designed and implemented a **12-layer Transformer Encoder** from scratch (without using `torch.nn.Transformer`), featuring Multi-Head Self-Attention (MHSA) and Sinusoidal Positional Encodings.
- **Advanced Optimization**:
    - Engineered a robust training pipeline utilizing **AdamW** (decoupled weight decay) and **Label Smoothing** ($\epsilon=0.05$) to mitigate overfitting in high-dimensional spaces ($d_{model}=512$).
- **Theoretical Analysis**:
    - Analyzed the trade-off between the **inductive bias** of RNNs (sequential invariance) and the **global receptive field** of Transformers.

## 🛠️ Model Architectures

### 1. Recurrent Baselines (RNN/LSTM/GRU)
- **Sequential Processing**: Models input $x_t$ based on hidden state $h_{t-1}$.
- **Gating Mechanisms**: Utilized LSTM and GRU cells to regulate information flow and alleviate the vanishing gradient problem.
- **Bidirectionality**: Implemented Bi-LSTM to capture both past and future contexts.

### 2. Transformer (Self-Attention) [Image of Transformer model architecture]
- **Structure**: 12-Layer Stacked Encoder.
- **Attention Mechanism**: 8-Head Self-Attention allowing parallel computation and global context aggregation.
- **Positional Encoding**: Sinusoidal injection to retain sequence order information in the absence of recurrence.
- **Normalization**: Pre-LayerNorm configuration for training stability.

## 📊 Experimental Results

Experiments were conducted on text classification tasks requiring the understanding of global semantic context.

| Model Architecture | Test Accuracy | Convergence Speed | Parameters |
| :--- | :---: | :---: | :---: |
| **Transformer (Ours)** | **98.51%** | **Fast (Parallelizable)** | **High** |
| Bi-LSTM | 95.45% | Slow (Sequential) | Medium |
| GRU | 94.80% | Medium | Medium |
| Vanilla RNN | < 60.0% | Unstable | Low |

**Key Finding:** The Transformer model outperformed the best recurrent baseline (Bi-LSTM) by a significant margin (**+3.06%**), validating the hypothesis that self-attention mechanisms are superior at modeling global dependencies in long sequences.

## 💻 Tech Stack & Hyperparameters

- **Framework**: PyTorch
- **Language**: Python 3.x
- **Key Hyperparameters (Best Model)**:
    - `Embedding Dimension`: 512
    - `Attention Heads`: 8
    - `Feed-Forward Dimension`: 1024
    - `Dropout`: 0.15
    - `Optimizer`: AdamW (`lr=9e-5`, `weight_decay=1e-2`)
    - `Loss Function`: CrossEntropyLoss with `Label Smoothing`

## 📂 Repository Structure
## 🔧 How to Run

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YourUsername/Capturing-Long-Range-Dependencies.git](https://github.com/YourUsername/Capturing-Long-Range-Dependencies.git)
   cd Capturing-Long-Range-Dependencies
   pip install torch numpy matplotlib
   python train.py --model transformer --epochs 30 --batch_size 64
