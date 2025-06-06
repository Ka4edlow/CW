import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("email_classifier_model.h5")

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)["encoder"]

subject = input("Введіть тему листа: ")
message = input("Введіть текст листа: ")
text = subject + " " + message

X_input = vectorizer.transform([text]).toarray()

prediction = model.predict(X_input)[0]

spam_prob = prediction[1]
predicted_label_num = 1 if spam_prob > 0.5 else 0

predicted_label_text = "Спам" if predicted_label_num == 1 else "Не спам"

print("\nТема:", subject)
print("Текст:", message)
print("Класифікація (число):", predicted_label_num)
print("Класифікація (текст):", predicted_label_text)
print(f"Ймовірність спаму: {spam_prob:.4f}")
print(f"Повні ймовірності: {prediction}")