# MLP-XOR-Classifier
Multilayer Perceptron (MLP) from scratch in NumPy for XOR classification

# MLP from Scratch: Solving the XOR Classification Task

This project implements a 2-2-1 Multilayer Perceptron (MLP) from scratch using NumPy to solve the XOR problem. No high-level libraries (like TensorFlow or PyTorch) are used.

## 🔧 Features

- 2 input neurons, 2 hidden neurons (Sigmoid/ReLU), 1 output (Sigmoid)
- Manual backpropagation using Mean Squared Error (MSE)
- Learning rate, activation function, and epoch tunable
- Evaluation metrics implemented from scratch:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve

## 📊 Results

The model is trained on the XOR dataset and achieves high performance with correct tuning.

## ⚙️ Configurable Parameters

- Learning rate: 0.01, 0.1, 0.5
- Epochs: 500, 1000, 5000
- Activation: `sigmoid` or `relu`

## 📁 Files

- `MLP_XOR_Classifier.ipynb` – Google Colab notebook
- `README.md` – Project description

## 🚀 Run It

Run the notebook directly in [Google Colab](https://colab.research.google.com/) or download and use Jupyter Notebook.

---

### 🧠 Theory

### MLP Equation

For layer \( l \):
- \( Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]} \)
- \( A^{[l]} = \sigma(Z^{[l]}) \)

### Evaluation Metric Formulas

- **Accuracy** = \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision** = \( \frac{TP}{TP + FP} \)
- **Recall** = \( \frac{TP}{TP + FN} \)
- **F1 Score** = \( 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} \)

---




