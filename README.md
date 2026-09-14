# Neural Networks

This project focuses on the fundamentals of neural networks and their implementation from scratch.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

Run the notebook or Python scripts after installing the dependencies.

## What was done

* Prepared a **time-based train/validation/test split** for the Don'tGetKicked dataset to prevent temporal data leakage.
* Preprocessed numerical and categorical features, including categorical encoding fitted only on the training data.
* Implemented a **single-hidden-layer MLP from scratch using NumPy**.
* Implemented:

  * forward propagation
  * binary cross-entropy loss
  * backpropagation
  * gradient computation
  * mini-batch training
  * SGD optimization
  * probability and class prediction
* Experimented with different **activation functions**:

  * Sigmoid
  * ReLU
  * Cosine
* Tuned the network architecture and training procedure to improve validation performance.
* Compared the custom MLP with **scikit-learn's MLPClassifier**.
* Implemented the same architecture using **PyTorch** and compared its performance with the from-scratch implementation.
* Evaluated models using the **Gini coefficient** on training, validation, and test sets.
* Analyzed the difference between validation and test performance to assess **overfitting and generalization**.
* Implemented and experimented with the **Adam optimizer**.

## Main techniques

`Neural Networks` · `MLP` · `Backpropagation` · `Gradient Descent` · `SGD` · `Adam` · `Activation Functions` · `Binary Cross-Entropy` · `NumPy` · `PyTorch`

## Dataset

* Don'tGetKicked — used-car auction dataset from Kaggle
