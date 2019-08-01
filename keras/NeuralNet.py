from keras import datasets, models
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class NeuralNet(models.Sequential):
    def __init__(self, n_in, n_h, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None

    def step_1(self):
        self.add(Dense(self.n_h, activation='relu', input_shape=(self.n_in, )))
        self.add(Dense(self.n_out, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        self.history = self.fit(self.x_train, self.y_train, epochs=5, batch_size=100, validation_split=0.2)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)

        L, W, H = x_train.shape

        x_train = x_train.reshape(-1, W * H)
        x_test = x_test.reshape(-1, W * H)

        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0

    def show_performance(self):
        performance_test = self.evaluate(self.x_test, self.y_test, batch_size=100)
        print('Test Loss and Accuracy -> ', performance_test)

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'test'], loc=0)
        plt.show()

    def plot_acc(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'test'], loc=0)
        plt.show()
