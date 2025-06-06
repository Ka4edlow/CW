import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(labels, num_classes):
    result = np.zeros((num_classes, len(labels)))
    result[labels, np.arange(len(labels))] = 1
    return result

class CIFAR10Loader:
    def __init__(self, data_path):
        self.data_path = data_path

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='latin1')

    def load_batch(self, filename):
        batch = self._unpickle(filename)
        images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = batch['labels']
        return images, labels

    def load_data(self):
        x_train, y_train = [], []
        for i in range(1, 6):
            images, labels = self.load_batch(os.path.join(self.data_path, f'data_batch_{i}'))
            x_train.append(images)
            y_train.extend(labels)
        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        x_test, y_test = self.load_batch(os.path.join(self.data_path, 'test_batch'))
        return x_train, y_train, np.array(x_test), np.array(y_test)

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def update(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))
        self.optimizer = AdamOptimizer([self.W1, self.b1, self.W2, self.b2], lr)

    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return (z > 0).astype(float)
    def softmax(self, z): 
        e = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e / np.sum(e, axis=0, keepdims=True)

    def forward(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_hat, Y):
        return -np.sum(Y * np.log(Y_hat + 1e-9)) / Y.shape[1]

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        dW2 = dZ2 @ self.A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = dZ1 @ X.T / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.optimizer.update([dW1, db1, dW2, db2])

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.W1, self.b1 = data['W1'], data['b1']
            self.W2, self.b2 = data['W2'], data['b2']

def train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size=64, patience=5, save_path=None):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[1])
        X_train, Y_train = X_train[:, indices], Y_train[:, indices]

        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_train[:, i:i + batch_size]
            Y_batch = Y_train[:, i:i + batch_size]
            Y_hat = model.forward(X_batch)
            model.backward(X_batch, Y_batch)

        val_pred = model.forward(X_val)
        val_loss = model.compute_loss(val_pred, Y_val)
        val_acc = np.mean(np.argmax(val_pred, axis=0) == np.argmax(Y_val, axis=0))
        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                model.save(save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                if save_path:
                    model.load(save_path)
                break

if __name__ == "__main__":
    loader = CIFAR10Loader('cifar-10-batches-py')
    x_train, y_train, x_test, y_test = loader.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    val_split = int(0.8 * x_train.shape[0])
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]

    X_train = x_train.reshape(x_train.shape[0], -1).T
    X_val = x_val.reshape(x_val.shape[0], -1).T
    X_test = x_test.reshape(x_test.shape[0], -1).T

    Y_train = one_hot_encode(y_train, 10)
    Y_val = one_hot_encode(y_val, 10)
    Y_test = one_hot_encode(y_test, 10)

    nn = NeuralNetwork(input_dim=3072, hidden_dim=2048, output_dim=10, lr=0.001)
    train_model(nn, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=128, patience=10, save_path='best_model.pkl')
    nn.load('best_model.pkl')

    preds = nn.predict(X_test)
    test_acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    class_labels = ['літак', 'автомобіль', 'птах', 'кіт', 'олень', 
                    'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']
    sample_idx = np.random.choice(x_test.shape[0], 10, replace=False)
    sample_images = x_test[sample_idx]
    sample_labels = y_test[sample_idx]
    sample_preds = nn.predict(X_test[:, sample_idx])

    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(sample_idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Прогноз: {class_labels[sample_preds[i]]}\nІстина: {class_labels[sample_labels[i]]}", fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
