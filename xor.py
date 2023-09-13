import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers) -1
        self.W = []
        self.B = []
        self.O = []
        self.A = []
        self.dW = []
        self.dB = []

        for i in range(self.length):
            self.W.append(np.random.randn(layers[i+1], layers[i]))
            self.B.append(np.random.randn(layers[i+1], 1))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, x):
        self.O = []
        self.A = [x]

        for i in range(self.length):
            self.O.append(np.dot(self.W[i], x) + self.B[i])
            self.A.append(self.sigmoid(self.O[i]))
        
        return self.A[self.length]

    def loss(self, predict, y):
        return (-(y * np.log(predict)) - ((1 - y) * np.log(1 - predict)))

    def backward(self, predict, y, learning_rate):
        self.dW = []
        self.dB = []
        delta = predict - y

        for i in reversed(range(self.length)):
            self.dW.append(np.dot(delta, self.A[i].T))
            self.dB.append(np.sum(delta, axis=0, keepdims=True))
            delta = np.multiply(np.dot(self.W[i].T, delta), self.A[i-1] * (1- self.A[i-1]))

        self.dW.reverse()
        self.dB.reverse()

        for i in range(self.length):
            self.W[i] = self.W[i] - self.dW[i] * learning_rate
            self.B[i] = self.B[i] - self.dB[i] * learning_rate
    
    def train(self, x, y, iterations, learning_rate, display):
        losses = []
        for i in range(iterations):
            predict = self.forward(x)
            self.backward(predict, y, learning_rate)
            loss = self.loss(predict, y)
            losses.append(loss)
        
        if display:
            plt.plot(range(iterations), losses)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.show()

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

nn = NeuralNetwork([2, 2, 1])
nn.train(X, Y, 5000, learning_rate=0.1, display=False)

print(nn.forward(X))