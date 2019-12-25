from keras.models import load_model,Sequential
import numpy as np
from PIL import Image
import json
import os

class Detector:
    image_shape = (66, 66, 3)

    @staticmethod
    def load_model() -> Sequential:
        return load_model("12306.h5")

    def __init__(self):
        self.model = Detector.load_model()
        with open('key_map.json', 'r') as fd:
            self.labels = json.load(fd)


    @staticmethod
    def get_array(image):
        return np.asarray(image.resize((Detector.image_shape[0], Detector.image_shape[1])))

    def predict_path(self, image_path):
        return self.predict(Image.open(image_path))

    def predict_dir(self, dir):
        images = []
        for item in os.listdir(dir):
            images.append(Image.open(dir + "/" + item))
        return self.predict_batch(images)


    def predict_batch(self,images):
        images = [x.resize((Detector.image_shape[0], Detector.image_shape[1])) for x in images]
        batch = np.asarray([np.asarray(x) for x in images])
        result = self.model.predict(batch)
        result_arr = []
        for x in result:
            idx = np.argmax(x)
            result_arr.append(self.labels[str(idx)])
        return result_arr

    def predict(self,image):
        input_x = self.get_array(image)
        input_x = input_x[np.newaxis,:]
        result =  self.model.predict(input_x)
        result_arr = []
        for x in result:
            idx = np.argmax(x)
            result_arr.append(self.labels[str(idx)])
        return result_arr

xx = Detector()
print(xx.predict_dir("archive/中国结"))


