"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import model_from_json
from keras.optimizers import Adam
import os
import numpy as np
import cv2

np.random.seed(777)  # reproducibility

CHECK_POINT_DIR = './RiderEnvironment/mnist_model_custom'
TRAIN_DATA_DIR = './RiderEnvironment/score_train_data'

class ScoreModel:
    def __init__(self):
        self.load()


    @staticmethod
    def allfiles(path):
        names = []
        for root, dirs, files in os.walk(path):
            for file in files:
                names.append(file.split('.')[0])

        return names

    @staticmethod
    def to_binary(img):
        _, threshold = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return np.array(threshold)


    def load_data(self):
        x = []
        y = []

        for i in range(10):
            path = "{}/{}".format(TRAIN_DATA_DIR, i)
            names = self.allfiles(path)
            for name in names:
                img = cv2.imread("{}/{}/{}.png".format(TRAIN_DATA_DIR, i, name), cv2.IMREAD_GRAYSCALE)
                img = self.to_binary(img)
                x.append(img.reshape(28, 28, 1))
                y_one_hot = np.zeros(10)
                y_one_hot[i] = 1
                y.append(y_one_hot)

        print(np.shape(x))
        print(np.shape(y))

        return np.array(x), np.array(y)

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_save(self):
        self.model = self.build_model()
        x, y = self.load_data()
        self.model.fit(np.array(x), np.array(y), batch_size=512 , epochs=50)

        self.save()

    def save(self):
        json = self.model.to_json()
        with open(CHECK_POINT_DIR+'/model.json', 'w') as file:
            file.write(json)

        self.model.save_weights(CHECK_POINT_DIR+'/model.h5')

    def load(self):
        if os.path.exists(CHECK_POINT_DIR+'/model.json') and os.path.exists(CHECK_POINT_DIR+'/model.h5'):
            with open(CHECK_POINT_DIR+'/model.json', 'r') as file:
                json = file.read()
                self.model = model_from_json(json)
                self.model.load_weights(CHECK_POINT_DIR+'/model.h5')

    def predict(self, img):
        img = img.reshape(1, 28, 28, 1)
        result = np.argmax(self.model.predict_on_batch(img)).flatten()[0]
        return result

if __name__ == '__main__':
    CHECK_POINT_DIR = './mnist_model_custom'
    TRAIN_DATA_DIR = './score_train_data'
    model = ScoreModel()
    model.train_save()