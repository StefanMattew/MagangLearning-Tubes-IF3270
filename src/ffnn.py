import numpy as np

class Activation:
    @staticmethod
    def relu(x, derivative=False):
        if derivative: return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        if derivative: return sig * (1 - sig)
        return sig

    @staticmethod
    def tanh(x, derivative=False):
        tanh = np.tanh(x)
        if derivative: return 1 - tanh ** 2
        return tanh
    
class Layer:
    def __init__ (self, input_size, output_size, activation, init_method= 'random', seed=None, lower=-0.5, upper=0.5, mean=0, var=1 ):
        
        if seed is not None:
            np.random.seed(seed)

        if init_method == 'zero':
            self.weights = np.zeros((input_size, output_size))
        if init_method == 'uniform':
            self.weights = np.random.uniform(lower, upper, (input_size, output_size))
        else :# init_method == 'random_normal'
            sd = np.sqrt(var) # karna random normal pakenya standar deviasi

            self.weights = np.random.normal(mean, sd, (input_size, output_size))

        self.bias = np.zeros((1, output_size))
        self.activation_name = activation
        self.input = None 
        self.z = None

class FFNN:
    def __init__(self, loss_function='BCE'):
        self.layers = []
        self.loss_function = loss_function

    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self, x_batch):
        input = x_batch

        for layer in self.layers:
            layer.input = input
            layer.z = np.dot(input, layer.weights) + layer.bias
            activation_func = getattr(Activation, layer.activation_name)
            input = activation_func(layer.z)
        return input

    

    def backward():
        pass

    def show_weights(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} Weights:\n{layer.weights}\nBias:\n{layer.bias}\n")
        #fungsi nampilin distribusi bobot

    def show_gradient(self):
        pass

    def save(self, filename):
        with open(filename, 'w') as f:
            for i, layer in enumerate(self.layers):
                f.write(f"Layer {i+1} Weights:\n{layer.weights}\nBias:\n{layer.bias}\n")

        print (f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'r') as f:
            model = f.read()
            # perlu cek dulu format
        print(f"Model loaded from {filename}:")


    def compute_loss(self, y_true, y_pred):
        # dibikin class ato langsung fungsi
        pass

    def fit(self):
        pass

 