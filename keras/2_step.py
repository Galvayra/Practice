from Practice.keras.NeuralNet import NeuralNet

Nin = 0
Nh = [100, 50]
number_of_class = 10
Nout = number_of_class

if __name__ == '__main__':
    nn = NeuralNet(Nin, Nh, Nout)
    nn.load_data_cifar()
    nn.step_2()
    nn.show_performance()
    nn.plot_loss()
    nn.plot_acc()
