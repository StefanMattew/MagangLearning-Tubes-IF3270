import numpy as np
import pickle

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

    @staticmethod
    def binary_step(x, derivative=False):
        if derivative: 
            return 0
        return np.where(x >= 0, 1, 0)
    
    @staticmethod
    def softsign(x, derivative=False):
        if derivative:
            return 1 / (1 + np.abs(x)) ** 2
        return x / (1 + np.abs(x))
    
    @staticmethod
    def leaky_relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0.01)
        return np.where(x > 0, x, 0.01 * x)

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
    def __init__(self, input_size, output_size, activation, init_method='random_normal', seed=None, lower=-0.5, upper=0.5, mean=0, var=1):
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.seed = seed
        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.var = var

        self.weights = None
        self.initialize_weights()

        self.bias = np.zeros((1, output_size))
        self.activation_name = activation
        self.input = None 
        self.z = None

        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

        if self.seed is not None:
            self.initialize_weights()

    def initialize_weights(self, rng=None):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        elif rng is None:
            rng = np.random.default_rng()

        if self.init_method == 'zero':
            self.weights = np.zeros((self.input_size, self.output_size))

        elif self.init_method == 'uniform':
            self.weights = rng.uniform(self.lower, self.upper, (self.input_size, self.output_size))

        elif self.init_method == 'random_normal':
            sd = np.sqrt(self.var)
            self.weights = rng.normal(self.mean, sd, (self.input_size, self.output_size))

        elif self.init_method == 'xavier':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = rng.uniform(-limit, limit, (self.input_size, self.output_size))

        elif self.init_method == 'he':
            std = np.sqrt(2 / self.input_size)
            self.weights = rng.normal(0, std, (self.input_size, self.output_size))

        else: # default random_normal
            self.weights = rng.uniform(self.lower, self.upper, (self.input_size, self.output_size))

        self.dweights = np.zeros_like(self.weights)

class FFNN:
    def __init__(self, loss_function='binary_cross_entropy', seed=None):
        self.layers = []
        self.loss_function = loss_function
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def add_layer(self, layer):
        if layer.seed is None:
            layer.initialize_weights(rng=self.rng)
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
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
        print (f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)

        print(f"Model loaded from {filename}:")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, learning_rate=0.01, epochs=100, l1_lambda=0.0, l2_lambda=0.0, verbose= 1):
        history = {
            'train_loss': [],
            'val_loss': []
        }

        n= X_train.shape[0]
        
        for epoch in range(epochs):
            # shuffle data
            indices = np.arange(n)
            self.rng.shuffle(indices)

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
            
            avg_train_loss = epoch_loss / n
            history['train_loss'].append(avg_train_loss)

            # Count validation loss
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                history['val_loss'].append(val_loss)

            if verbose == 1 and ((epoch + 1) % 10 == 0 or epoch == 0):
                status = f"Epoch {epoch + 1}/{epochs}.  Train Loss: {avg_train_loss:.4f}"
                
                if X_val is not None and y_val is not None:
                    status += f", Val Loss: {val_loss:.4f}"
                
                print(status)

        return history
    
class RMSNorm:
    def __init__(self, size, eps=1e-8):
        self.gamma = np.ones((1, size))
        self.beta = np.zeros((1, size))
        self.eps = eps
        
        self.x = None
        self.x_norm = None
        self.rms = None

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm + self.beta

    def backward(self, dout, learning_rate):
        _, D = dout.shape
        
        dgamma = np.sum(dout * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        dx_norm = dout * self.gamma
        drms = np.sum(dx_norm * self.x * (-1.0 / (self.rms**2)), axis=-1, keepdims=True)
        dx = (dx_norm / self.rms) + (drms * self.x / (D * self.rms))
        
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx