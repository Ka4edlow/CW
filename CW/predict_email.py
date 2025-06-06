import pickle
import numpy as np
from mlp_model import MLP

with open("mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_data = pickle.load(f)
    label_encoder = label_data["encoder"]

subject = input("Введіть тему листа: ")
message = input("Введіть текст листа: ")
text = subject + " " + message

X_input = vectorizer.transform([text]).toarray()

probs = model.forward(X_input)[0]

spam_prob = probs[1]
predicted_label_num = 1 if spam_prob > 0.5 else 0
predicted_label_text = "Спам" if predicted_label_num == 1 else "Не спам"

print("\nТема:", subject)
print("Текст:", message)
print("Класифікація (число):", predicted_label_num)
print("Класифікація (текст):", predicted_label_text)
print(f"Ймовірність спаму: {spam_prob:.4f}")
print(f"Повні ймовірності: {probs}")