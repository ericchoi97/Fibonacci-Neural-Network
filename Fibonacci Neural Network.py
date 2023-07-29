import numpy as np
import matplotlib.pyplot as plt

# Activate ReLU
def relu(x):
    return np.maximum(0, x)

# derivative for backpropagation
def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# Mean Squared Error Loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Training Data
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2], [4], [6], [8], [10]])

# Weights
weights0 = 2 * np.random.random((1, 5)) - 1
weights1 = 2 * np.random.random((5, 1)) - 1

# Learning rate
lr = 0.01

# Loss storage
losses = []

# ANN Training
for i in range(10000):  # Increase number of iterations
    # Feedforward
    layer0 = x
    layer1 = relu(np.dot(layer0, weights0))
    layer2 = np.dot(layer1, weights1)

    # Backpropagation
    layer2_error = y - layer2
    layer2_delta = layer2_error * lr

    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * relu_derivative(layer1)

    # New weights
    weights1 += layer1.T.dot(layer2_delta)
    weights0 += layer0.T.dot(layer1_delta)

    # Loss & Append
    loss = mse_loss(y, layer2)
    losses.append(loss)

# Loss Plot
plt.plot(losses)
plt.xlabel('Training')
plt.ylabel('Loss')
plt.show()

# Prediction
def predict(numbers):
    for i in numbers:
        layer0 = np.array([[i]])
        layer1 = relu(np.dot(layer0, weights0))
        layer2 = np.dot(layer1, weights1)
        print(f"Input: {i}, Output: {layer2[0][0]}")

# Fibonacci Series Prediction
numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
predict(numbers)
