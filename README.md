# IMDb Sentiment Analysis using a Custom Transformer Encoder

## Overview

This project implements a **sentiment analysis model for movie reviews** using the **IMDb dataset**.  
The goal is to classify reviews into **positive** or **negative** sentiment using a **Transformer-based architecture implemented from scratch in PyTorch**.

The project focuses on building and training a **custom Transformer encoder** for text classification, including data preprocessing, subword tokenization, model training, evaluation, and inference.

---

# Dataset

The model is trained on the **IMDb Movie Reviews Dataset**, which contains labeled movie reviews.

Dataset characteristics:

- **50,000 movie reviews**
- **Binary sentiment labels:** positive / negative
- Balanced dataset

Each sample contains:

| Column | Description |
|------|-------------|
| review | Movie review text |
| sentiment | Sentiment label |

---

# Project Pipeline

## 1. Text Preprocessing

Raw reviews are cleaned using several preprocessing steps:

- HTML tag removal
- Unicode normalization
- Contraction expansion
- Lowercasing
- Removing unnecessary characters
- Whitespace normalization

These steps standardize the text before tokenization.

---

## 2. Subword Tokenization (BPE)

The project uses **Byte Pair Encoding (BPE)** for tokenization.

Tokenizer configuration:

- Vocabulary size: **25,000**
- Special tokens:
  - `<PAD>`
  - `<UNK>`

Each review is converted into a **sequence of token IDs**.

---

## 3. Text Vectorization

Reviews are converted into numerical sequences for training:

- Maximum sequence length: **512 tokens**
- Padding applied using `<PAD>`
- Longer sequences are truncated

---

# Transformer Model

The sentiment classifier is based on a **custom Transformer encoder architecture**.

Key components:

- Token embedding layer
- Sinusoidal positional encoding
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Dropout regularization
- Mean pooling for sequence representation

The pooled representation of the sequence is passed through a **fully connected classification layer** to predict sentiment.

---

# Training

Training configuration:

- Framework: **PyTorch**
- Loss function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Learning rate scheduling
- Gradient clipping
- Early stopping based on validation loss

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

# Inference

The project includes a function for predicting the sentiment of new reviews.

Example:

```python
predict_sentiment("This movie was amazing. The story and acting were fantastic!")
```
You can access the online project via [this link](https://colab.research.google.com/drive/1ISXzES-kAHqrTaPSU9MjRNtjDf6wfzWj).

