import numpy as np
import torch
import torch.nn as nn

class my_MLP:
    def __init__(self, n_hidden=100, activation='relu', batch_size=64, random_state=21, solver='adam', max_iter=200, learning_rate=0.001, tol=1e-4, n_iter_no_change=10, early_stopping=True, validation_fraction=0.1):
        self.n_hidden = n_hidden
        self.activation, self.derivative = self._activation_derivative(activation)
        self.batch_size = batch_size
        self.random = np.random.default_rng(seed=random_state)
        self.solver = solver
        if solver == 'adam':
            self.t = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.threshold = 0.5
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction

    def _initialize_weights(self, n_feat):
        self.w1 = self.random.normal(0, 0.01, (n_feat, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = self.random.normal(0, 0.01, (self.n_hidden, 1))
        self.b2 = np.zeros(1)

        self.params = [self.w1, self.b1, self.w2, self.b2]

        if self.solver == 'adam':
            self.m = [np.zeros_like(p) for p in self.params]
            self.v = [np.zeros_like(p) for p in self.params]
            self.t = 0

    def _sigmoid(self, a):
        a = np.clip(a, -500, 500)
        return 1 / (1 + np.exp(-a))

    def _sigmoid_derivative(self, a):
        return self._sigmoid(a) * (1 - self._sigmoid(a))

    def _activation_derivative(self, activ):
        if activ == 'relu':
            activation = lambda x: np.maximum(0, x)
            derivative = lambda x: np.where(x > 0, 1, 0)
        elif activ == 'sigmoid':
            activation = self._sigmoid
            derivative = self._sigmoid_derivative
        elif activ == 'tanh':
            activation = np.tanh
            derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activ == 'cosine':
            activation = np.cos
            derivative = lambda x: -np.sin(x)
        else:
            raise ValueError(f"Unknown activation function {activ}")

        return activation, derivative

    def _forward(self, X):
        Z1 = X @ self.w1 + self.b1
        A1 = self.activation(Z1)
        Z2 = A1 @ self.w2 + self.b2
        A2 = self._sigmoid(Z2)

        return Z1, A1, Z2, A2

    def _compute_loss(self, p, y):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return - np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _backward(self, Z1, A1, Z2, A2, X, y):
        # dA2 = (-y / A2 + (1 - y) / (1 - A2)) / y.shape[0]
        # dZ2 = dA2 * self._sigmoid_derivative(Z2)
        dZ2 = (A2 - y) / y.shape[0] # numerical more stable for BCE 
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.w2.T
        dZ1 = dA1 * self.derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        grads = [dW1, db1, dW2, db2]
        return grads

    def _update_params(self, grads):

        if self.solver == 'sgd':
            for param, grad in zip(self.params, grads):
                param -= self.learning_rate * grad

        elif self.solver == "adam":
            self.t += 1

            for i in range(len(self.params)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                self.params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        else:
            raise ValueError(f"Unknown solver {self.solver}")

    def predict_proba(self, X):
        _, _, _, P1 = self._forward(X)
        return np.column_stack((1 - P1, P1))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y[:, None]

        n_samples = X.shape[0]
        n_feat = X.shape[1]

        if self.early_stopping:
            indices = self.random.permutation(n_samples)

            n_valid = int(n_samples * self.validation_fraction)

            valid_indices = indices[:n_valid]
            train_indices = indices[n_valid:]

            X_valid = X[valid_indices]
            y_valid = y[valid_indices]

            X_train = X[train_indices]
            y_train = y[train_indices]

        else:
            X_train = X
            y_train = y

        self._initialize_weights(n_feat)

        n_train = X_train.shape[0] 
        n_batch = int(np.ceil(n_train / self.batch_size))

        iteration_with_no_changing = 0

        if self.early_stopping:
            best_valid_loss = np.inf
            best_params = None

        for epoch in range(self.max_iter):

            indices = self.random.permutation(n_train)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(n_batch):
                X_batch = X_shuffled[i * self.batch_size : (i + 1) * self.batch_size]
                y_batch = y_shuffled[i * self.batch_size : (i + 1) * self.batch_size]


                Z1, A1, Z2, P1 = self._forward(X_batch)
                grads = self._backward(Z1, A1, Z2, P1, X_batch, y_batch)
                self._update_params(grads)

            if self.early_stopping:

                valid_proba = self.predict_proba(X_valid)[:, -1]

                valid_loss = self._compute_loss(valid_proba, y_valid)

                if valid_loss < best_valid_loss - self.tol:

                    best_valid_loss = valid_loss
                    iteration_with_no_changing = 0

                    best_params = [param.copy() for param in self.params]

                else:
                    iteration_with_no_changing += 1

                if iteration_with_no_changing >= self.n_iter_no_change:
                    break

        if self.early_stopping and best_params is not None:
            for param, best_param in zip(self.params, best_params):
                param[:] = best_param
            
        return self

    def predict(self, X):
        X = np.asarray(X)
        proba = self.predict_proba(X)[:, 1]

        return np.where(proba >= self.threshold, 1, 0)

class py_MLP(nn.Module):
    def __init__(self, n_features, hidden_layers=[100], activation='relu', optimizer='adam', learning_rate=0.001, validation_fraction=0.1, patience=10, epochs=300, batch_size=64):
        super().__init__()

        self.validation_fraction = validation_fraction
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size

        if activation == 'relu':
            activation_function = nn.ReLU()
        elif activation == 'sigmoid':
            activation_function = nn.Sigmoid()
        elif activation == 'tanh':
            activation_function = nn.Tanh()
        else:
            raise ValueError('Unknown activation')

        layers = []
        input_size = n_features

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_function)
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Unknown optimizer')

        self.criterion = nn.BCELoss()

    def forward(self, X):
        return self.network(X)

    def fit(self, X, y):

        n_valid = int(len(X) * self.validation_fraction)

        indices = torch.randperm(len(X))

        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_valid = X[valid_indices]
        y_valid = y[valid_indices]

        best_valid_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):

            self.train()

            permutation = torch.randperm(len(X_train))

            for start in range(0, len(X_train), self.batch_size):

                batch_indices = permutation[
                    start:start + self.batch_size
                ]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                self.optimizer.zero_grad()

                predictions = self(X_batch)

                train_loss = self.criterion(predictions, y_batch)

                train_loss.backward()

                self.optimizer.step()

            self.eval()

            with torch.no_grad():
                valid_predictions = self(X_valid)
                valid_loss = self.criterion(valid_predictions, y_valid)

            if valid_loss < best_valid_loss:
                
                best_valid_loss = valid_loss
                best_state = self.state_dict().copy()

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        self.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.eval()

        with torch.no_grad():
            predictions = self(X).numpy()

        return np.column_stack((1 - predictions, predictions))

    def predict(self, X):

        proba = self.predict_proba(X)[:, 1]

        return np.where(proba >= 0.5, 1, 0)