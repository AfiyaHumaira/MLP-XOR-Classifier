# MLP-XOR-Classifier
Multilayer Perceptron (MLP) from scratch in NumPy for XOR classification

# 🧠 MLP from Scratch: Solving the XOR Classification Task

This project implements a **Multilayer Perceptron (MLP)** from scratch using **NumPy** to solve the **XOR classification problem** — a classic example of a non-linearly separable task. The model is trained using manual backpropagation and weight updates without using any deep learning libraries.

---

## ✅ Features

- Custom 2-2-1 MLP architecture
- Activation options: **Sigmoid** or **ReLU**
- Manual **backpropagation** with Mean Squared Error (MSE) loss
- Tunable:
  - Learning rate: `0.01`, `0.1`, `0.5`
  - Epochs: `500`, `1000`, `5000`
- Evaluation metrics implemented **from scratch**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve

---

## ⚙️ Configuration

These parameters can be changed in the code:

```python``
activation = 'sigmoid'     # Options: 'sigmoid' or 'relu'
learning_rate = 0.1        # Options: 0.01, 0.1, 0.5
epochs = 5000              # Options: 500, 1000, 5000



