import numpy as np
import neunet as nn

n = nn.NeuralNetwork([784, 18, 18, 10], ['identity', 'identity', 'identity', 'sigmoid'], 0, 0.01, 1)
input_data = np.loadtxt('train_set_input.txt')
result = np.loadtxt('train_set_output.txt')

n.train(input_data, result, 20, 0.01, progress_bar=True)

nn.save_weights(n)
