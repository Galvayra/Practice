# CNN for cifar
# from Practice.keras.Models import NeuralNet
from keras import datasets
from keras.backend import image_data_format
from Practice.keras.keraspp.aicnn import Machine

assert image_data_format() == 'channels_last'


class Step4(Machine):
    def __init__(self):
        (X, y), (_, _) = datasets.cifar10.load_data()
        super().__init__(X, y, nb_classes=10)


if __name__ == '__main__':
    m = Step4()
    m.run()
