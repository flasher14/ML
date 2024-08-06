import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.W1, self.B1 = np.random.randn(input_size, hidden_size),np.random.randn(1, hidden_size)
	self.W2, self.B2 = np.random.randn(hidden_size, output_size),np.random.randn(1, output_size)
 	self.lr = lr
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): return x * (1 - x)
    def train(self, X, y, epochs):
        for _ in range(epochs):
 	    H = self.sigmoid(X @ self.W1 + self.B1)
 	    O = self.sigmoid(H @ self.W2 + self.B2)
 	    O_delta = (y - O) * self.sigmoid_derivative(O)
 	    H_delta = (O_delta @ self.W2.T) * self.sigmoid_derivative(H)
 	    self.W2 += self.lr * H.T @ O_delta
 	    self.B2 += self.lr * np.sum(O_delta, axis=0, keepdims=True)
 	    self.W1 += self.lr * X.T @ H_delta
 	    self.B1 += self.lr * np.sum(H_delta, axis=0, keepdims=True)
    def predict(self, X):
	return self.sigmoid(self.sigmoid(X @ self.W1 + self.B1) @ self.W2 + self.B2)
X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(2, 3, 1, 0.1)
nn.train(X, y, 10000)
print("Predictions:\n", nn.predict(X))