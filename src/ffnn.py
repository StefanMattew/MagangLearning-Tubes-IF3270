import numpy as np

class Activation:
    @staticmethod
    def linear(x, derivative=False):
        if derivative: 
            return 1
        return x
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative: 
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        if derivative: 
            return sig * (1 - sig)
        return sig

    @staticmethod
    def tanh(x, derivative=False):
        tanh = np.tanh(x)
        if derivative: 
            return 1 - tanh ** 2
        return tanh

    @staticmethod
    def softmax(x, derivative=False):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        if derivative: 
            return softmax * (1 - softmax)
        return softmax

class Loss:
    @staticmethod
    def mse(y_true, y_pred, derivative=False):
        # y_true, y_pred = arr
        if derivative:
            return 2 * (y_pred - y_true) / y_true.shape[0]
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, derivative=False):
        # y_true, y_pred = arr
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, derivative=False):
        # y_true, y_pred = arr
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return -y_true / y_pred / y_true.shape[0]
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
class Layer:
    def __init__ (self, input_size, output_size, activation, init_method= 'random', seed=None, lower=-0.5, upper=0.5, mean=0, var=1 ):
        
        if seed is not None:
            np.random.seed(seed)

        if init_method == 'zero':
            self.weights = np.zeros((input_size, output_size))
        elif init_method == 'uniform':
            self.weights = np.random.uniform(lower, upper, (input_size, output_size))
        else :# init_method == 'random_normal'
            sd = np.sqrt(var) # karna random normal pakenya standar deviasi

            self.weights = np.random.normal(mean, sd, (input_size, output_size))

        self.bias = np.zeros((1, output_size))
        self.activation_name = activation
        self.input = None 
        self.z = None

        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

class FFNN:
    def __init__(self, loss_function='binary_cross_entropy'):
        self.layers = []
        self.loss_function = loss_function

    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self, x_batch):
        current_input = x_batch

        for layer in self.layers:
            layer.input = current_input
            layer.z = np.dot(current_input, layer.weights) + layer.bias
            
            activation_func = getattr(Activation, layer.activation_name)
            current_input = activation_func(layer.z)
        return current_input

    def compute_loss(self, y_true, y_pred, derivative=False):
        loss_func = getattr(Loss, self.loss_function)
        return loss_func(y_true, y_pred, derivative)
    
    def backward(self, y_true, y_pred, l1_lambda=0.0, l2_lambda=0.0):
        delta = self.compute_loss(y_true, y_pred, derivative=True)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation_func = getattr(Activation, layer.activation_name)
            d_activation= activation_func(layer.z, derivative=True)

            delta = delta * d_activation
            layer.dweights = np.dot(layer.input.T, delta)
            layer.dbias = np.sum(delta, axis=0, keepdims=True)

            if l1_lambda > 0:
                layer.dweights += l1_lambda * np.sign(layer.weights)
            if l2_lambda > 0:
                layer.dweights += l2_lambda * layer.weights

            if i > 0:
                delta = np.dot(delta, layer.weights.T)

    def show_weights(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} Weights:\n {layer.weights}\n Bias:\n{layer.bias}\n")
        #fungsi nampilin distribusi bobot

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.bias -= learning_rate * layer.dbias

    def show_gradient(self):
        pass

    def save(self, filename):
        with open(filename, 'w') as f:
            for i, layer in enumerate(self.layers):
                f.write(f"Layer {i+1} Weights:\n {layer.weights}\n Bias:\n{layer.bias}\n")

        print (f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'r') as f:
            model = f.read()
            # perlu cek dulu format
        print(f"Model loaded from {filename}:")
    
    def fit(self, X_train, y_train, batch_size=32, learning_rate=0.01, epochs=100, l1_lambda=0.0, l2_lambda=0.0, verbose= 1):
        history = []
        n= X_train.shape[0]
        
        for epoch in range(epochs):
            # shuffle data
            indices = np.arange(n)
            np.random.shuffle(indices)

            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0

            for i in range(0, n, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_pred = self.forward(X_batch)

                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss * X_batch.shape[0]

                self.backward(y_batch, y_pred, l1_lambda, l2_lambda)

                self.update_weights(learning_rate)
            
            avg_loss = epoch_loss / n
            history.append(avg_loss)

            if verbose == 1 and (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}.  Loss: {avg_loss:.4f}")

        return history
        

 