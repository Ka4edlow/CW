import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter

np.random.seed(42)

df = pd.read_csv("messages.csv")
df['text'] = df['subject'].fillna('') + ' ' + df['message'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(df['text']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

print("Баланс класів у вибірках:")
print("Весь датасет:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Баланс класів у тренуванні:", Counter(y_train))
print("Баланс класів у тесті:", Counter(y_test))


class MLP:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.001, dropout_rate=0.3):
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
        self.b2 = np.zeros(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size)
        self.b3 = np.zeros(output_size)

        self.train_mode = True

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def dropout(self, x):
        if self.train_mode:
            mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            return x * mask, mask
        else:
            return x, None

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.A1_drop, self.mask1 = self.dropout(self.A1)

        self.Z2 = self.A1_drop.dot(self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.A2_drop, self.mask2 = self.dropout(self.A2)

        self.Z3 = self.A2_drop.dot(self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m
        return loss

    def backward(self, X, Y_true):
        m = Y_true.shape[0]

        dZ3 = (self.A3 - Y_true) / m
        dW3 = self.A2_drop.T.dot(dZ3)
        db3 = np.sum(dZ3, axis=0)

        dA2_drop = dZ3.dot(self.W3.T)
        dA2 = dA2_drop * (self.mask2 if self.train_mode else 1)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = self.A1_drop.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0)

        dA1_drop = dZ2.dot(self.W2.T)
        dA1 = dA1_drop * (self.mask1 if self.train_mode else 1)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        self.train_mode = False
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, patience=7, model_path="mlp_best_model.pkl"):
        self.train_mode = True
        n_samples = X_train.shape[0]
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        best_weights = None
        accuracy_threshold = 0.90

        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                Y_batch = Y_train_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, Y_batch)

            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(train_pred, Y_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(Y_train, axis=1))

            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(val_pred, Y_val)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(Y_val, axis=1))

            print(f"Епоха {epoch + 1}/{epochs} | Втрата тренування: {train_loss:.4f} | Точність: {train_acc:.4f} | Валідація: {val_acc:.4f}")

            if val_acc >= accuracy_threshold:
                print(f"Досягнута точність {val_acc:.4f} ≥ {accuracy_threshold}. Збереження та зупинка навчання.")
                with open(model_path, "wb") as f:
                    pickle.dump(self, f)
                return

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_weights = {
                    'W1': self.W1.copy(), 'b1': self.b1.copy(),
                    'W2': self.W2.copy(), 'b2': self.b2.copy(),
                    'W3': self.W3.copy(), 'b3': self.b3.copy()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Перенавчання без покращення {patience} разів. Найкраща точність: {best_val_acc:.4f} (епоха {best_epoch+1})")
                break

        if best_weights:
            self.W1, self.b1 = best_weights['W1'], best_weights['b1']
            self.W2, self.b2 = best_weights['W2'], best_weights['b2']
            self.W3, self.b3 = best_weights['W3'], best_weights['b3']
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
            print("Збережена найкраща доступна модель (менше 90% точності).")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_onehot = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

mlp = MLP(input_size=2000, hidden1_size=256, hidden2_size=128, output_size=2, lr=0.001, dropout_rate=0.3)
mlp.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, patience=5, model_path="mlp_best_model.pkl")

with open("mlp_best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

y_pred = best_model.predict(X_test)
test_acc = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"\nТочність на тесті (найкраща модель): {test_acc:.4f}")

print("\nЙмовірності для перших 5 тестових зразків:")
print(mlp.forward(X_test[:5]))

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump({
        "encoder": label_encoder,
        "classes": label_encoder.classes_
    }, f)

with open("mlp_model.pkl", "wb") as f:
    pickle.dump(mlp, f)

print("Навчання завершено, модель та інструменти збережено.")