from sklearn import model_selection
from keras import datasets
import keras
from Practice.keras.keraspp import aiprt

assert keras.backend.image_data_format() == 'channels_last'


class Machine(aiprt.Machine_Generator):
    def __init__(self):
        (x_train, y_train), (_, _) = datasets.cifar10.load_data()
        _, X, _, y = model_selection.train_test_split(x_train, y_train, test_size=0.02)
        X = X.astype(float)

        # gen_param_dict = {'rotation_range': 10}

        super().__init__(X, y, nb_classes=10)


def main():
    m = Machine()
    m.run()


if __name__ == '__main__':
    main()
