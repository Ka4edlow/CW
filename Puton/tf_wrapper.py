import numpy as np

class TFModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, image_array):
        return self.model.predict(image_array)