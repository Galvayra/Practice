from Practice.keras.NeuralNet import NeuralNet

Nin = 784
Nh = 100
number_of_class = 10
Nout = number_of_class

if __name__ == '__main__':
    nn = NeuralNet(Nin, Nh, Nout)
    nn.load_data()
    nn.step_1()
    nn.show_performance()
    nn.plot_loss()
    nn.plot_acc()
