from keras import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D, Flatten
from keras.backend import *
from keras.callbacks import TensorBoard

class AE_Model:
    def __init__(self, height, width):
        self.build_model(height, width)

    def build_model(self, height, width):
        input_img = Input(shape=(height, width, 1))

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        print(encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def fit(self, img, height, width):
        img = np.array(img).reshape(1, height, width, 1)
        self.autoencoder.train_on_batch(img, img)
        return self.autoencoder.predict_on_batch(img).reshape(height, width)