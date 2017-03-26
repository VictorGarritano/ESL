import numpy as np 

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class neural_net:
    def __init__(self, eta, numrounds, hidden_nodes):
        self.numrounds = numrounds
        self.eta = eta
        self.hidden_nodes = hidden_nodes
        self.weights = None

    def initialize_weights(self, input_features, output_size):
        self.hidden_nodes.insert(0, input_features)
        self.hidden_nodes.append(output_size)
        self.weights = [np.random.normal(size=(x, y)) \
        for x, y in zip(self.hidden_nodes[:-1], self.hidden_nodes[1:])]

    def fit(self, X, y):
        X = np.c_[X, np.ones((X.shape[0], 1))] #  <-- bias trick
        samples, features = X.shape
        if len(y.shape) == 1: #  <-- in case of y.shape = (N,) 
            classes = 1
        else:
            _, classes = y.shape
        self.initialize_weights(features, classes)
        epoch = 0
        while epoch < self.numrounds:
            layer_outputs, activations, deltas = [], [], []
            activations.append(X)

            #forward pass
            for w in self.weights:
                z = activations[-1].dot(w)
                layer_outputs.append(z)
                activations.append(sigmoid(z))
            y_hat = activations[-1]
            loss = np.mean(0.5 * np.square(y - y_hat))
            if epoch % 1000 == 0:
                print (loss)

            #backward pass - Gradient Descent
            error = -(y - y_hat)
            delta = error * sigmoid_prime(layer_outputs[-1])
            deltas.append(activations[-2].T.dot(delta)) #   <-- gradient of the last weight matrix
            for i in range(1, len(self.weights)):
                delta = delta.dot(self.weights[-i].T) * sigmoid_prime(layer_outputs[-i-1])
                deltas.append(activations[-i-2].T.dot(delta))
            deltas = deltas[::-1]
            self.weights = [np.subtract(w, self.eta * d) for w, d in zip(self.weights, deltas)]
            epoch += 1
        # print (activations[-1])

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0],1))] #  <-- bias trick
        samples, _ = X.shape
        for w in self.weights:
            X = sigmoid(X.dot(w))
        X = [0.0 if sample < 0.5 else 1.0 for sample in X]
        return np.array(X)

# net = neural_net(eta=1.0, numrounds=60000, hidden_nodes=[10, 30, 30, 10])

# X = np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1]
#     ])
# y = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
#     ])
# net.fit(X, y)
