import os
import pickle
import numpy as np
import tensorflow as tf
from tf_wrapper import TFModelWrapper

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

def load_batch(filename):
    batch = unpickle(filename)
    images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return images, labels

def load_cifar10_data(data_path='cifar-10-batches-py'):
    x_train, y_train = [], []
    for i in range(1, 6):
        images, labels = load_batch(os.path.join(data_path, f'data_batch_{i}'))
        x_train.append(images)
        y_train.extend(labels)
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    x_test, y_test = load_batch(os.path.join(data_path, 'test_batch'))
    return x_train, y_train, np.array(x_test), np.array(y_test)

x_train, y_train, x_test, y_test = load_cifar10_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

val_split = int(0.8 * x_train.shape[0])
x_val, y_val = x_train[val_split:], y_train_cat[val_split:]
x_train, y_train = x_train[:val_split], y_train_cat[:val_split]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128, callbacks=[early_stop])

class TFModelWrapper:
    def __init__(self, model):
        self.model = model
        self.class_labels = ['літак', 'автомобіль', 'птах', 'кіт', 'олень',
                             'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

    def predict(self, image_array):
        preds = self.model.predict(image_array)
        return self.class_labels[np.argmax(preds)]

wrapper = TFModelWrapper(model)
with open('mlp_cifar10_best_model_tensorflow.pkl', 'wb') as f:
    pickle.dump(wrapper, f)

print("TensorFlow модель збережено як mlp_cifar10_best_model_tensorflow.pkl")
