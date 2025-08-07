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

```python
activation = 'sigmoid'     # Options: 'sigmoid' or 'relu'
learning_rate = 0.1        # Options: 0.01, 0.1, 0.5
epochs = 5000              # Options: 500, 1000, 5000
```
## 🔍 How It Works
The model is trained on the XOR dataset:
```python
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = [[0],
     [1],
     [1],
     [0]]
```

## 🧠 MLP Theory
- Architecture
    - Input Layer: 2 neurons
    - Hidden Layer: 2 neurons (Sigmoid or ReLU)
    - Output Layer: 1 neuron (Sigmoid)

- Forward Propagation
```python
Z1 = X · W1 + b1
A1 = activation(Z1)        # ReLU or Sigmoid
Z2 = A1 · W2 + b2
A2 = sigmoid(Z2)           # Final prediction
```
- Loss Function (Mean Squared Error)
 ```python
Loss = (1/n) * Σ (y_true - y_pred)^2
 ```
Where `n` is the number of samples. 
- Backpropagation
```python
dA2 = (y - A2) * sigmoid_derivative(A2)
dW2 = A1.T · dA2
db2 = sum of dA2 over all samples

dA1 = dA2 · W2.T * activation_derivative(A1)
dW1 = X.T · dA1
db1 = sum of dA1 over all samples
```
  - Weights and biases update:
 ```python
W2 += learning_rate * dW2
b2 += learning_rate * db2
W1 += learning_rate * dW1
b1 += learning_rate * db1
  ```  
## 📊 Evaluation Metrics (Implemented from Scratch)
```python
 Accuracy = (TP + TN) / Total
 Precision = TP / (TP + FP)
 Recall = TP / (TP + FN)
 F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```
- 🔸 ROC Curve
  A ROC (Receiver Operating Characteristic) curve is plotted using threshold values between 0 and 1 to evaluate the classifier's performance.

## ✅ Sample Output
```python
Expected:   [0 1 1 0]
Predicted:  [0 1 1 0]

Evaluation Metrics:
Accuracy:  1.00
Precision: 1.00
Recall:    1.00
F1 Score:  1.00
```
## 📁 Files
- ```MLP_XOR_Classifier.ipynb``` – Colab/Jupyter Notebook
- ```README.md``` – Project description and theory

## 🚀 How to Run
You can run this notebook using:
- [Google Colab](https://colab.research.google.com/)
- Jupyter Notebook (locally)
