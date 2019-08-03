from keras import datasets, models, backend
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class NeuralNet(models.Sequential):
    def __init__(self, n_in, n_h, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        self.input_shape_ = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None

    def load_data_mnist(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)

        _, W, H = x_train.shape
        self.n_in = W * H

        x_train = x_train.reshape(-1, self.n_in)
        x_test = x_test.reshape(-1, self.n_in)

        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0

        self.show_info()

    def load_data_cifar(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)

        _, W, H, C = x_train.shape

        self.n_in = W * H * C
        self.x_train = x_train.reshape(-1, self.n_in)
        self.x_test = x_test.reshape(-1, self.n_in)

        self.show_info()

    def load_data_mnist_for_cnn(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        _, W, H = x_train.shape

        if backend.image_data_format() == 'channels_first':
            print('channels_first')
            x_train = x_train.reshape(x_train.shape[0], 1, W, H)
            x_test = x_test.reshape(x_test.shape[0], 1, W, H)
            input_shape = (1, W, H)
        else:
            x_train = x_train.reshape(x_train.shape[0], W, H, 1)
            x_test = x_test.reshape(x_test.shape[0], W, H, 1)
            input_shape = (W, H, 1)

        self.y_train = np_utils.to_categorical(y_train, self.n_out)
        self.y_test = np_utils.to_categorical(y_test, self.n_out)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.input_shape_ = input_shape

        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0

        self.show_info()

    def load_data_cifar_for_cnn(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)

        _, W, H, C = x_train.shape

        self.input_shape_ = W * H * C
        self.x_train = x_train.reshape(-1, self.n_in)
        self.x_test = x_test.reshape(-1, self.n_in)

        self.show_info()

    def show_info(self):
        print("\n# of input nodes -", self.n_in, "\n\n")

    def training(self):
        print("\n\n\n\n\n======================= Training !! =======================")
        self.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        self.history = self.fit(self.x_train, self.y_train, epochs=5, batch_size=100, validation_split=0.2)

    def show_performance(self):
        performance_test = self.evaluate(self.x_test, self.y_test, batch_size=100)
        print('Test Loss and Accuracy -> ', performance_test)

    # ANN
    def step_1(self):
        self.add(Dense(self.n_h, activation='relu', input_shape=(self.n_in, )))
        self.add(Dense(self.n_out, activation='softmax'))

        self.training()

    # DNN
    def step_2(self):
        self.add(Dense(self.n_h[0], activation='relu', input_shape=(self.n_in, ), name="hidden_1"))
        self.add(Dropout(0.2))
        self.add(Dense(self.n_h[1], activation='relu', input_shape=(self.n_h[0], ), name="hidden_2"))
        self.add(Dropout(0.2))
        self.add(Dense(self.n_out, activation='softmax'))

        self.training()

    # CNN Mnist
    def step_3(self):
        self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape_))
        self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(self.n_out, activation='softmax'))

        self.training()

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
