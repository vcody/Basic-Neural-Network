import numpy as np

def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e ^ (-x))
    return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
    # y_true/y_pred: numpy arrays of same length
    return ((y_true - y_pred) ** 2).mean()

def derive_sigmoid(x):
    # Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid_function(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        # Weight inputs, add bias, use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid_function(total)
    
class NeuralNetwork:
    '''
    Neural Network with parameters:
    - input layer with two inputs
    - hidden layer with two neurons (h1, h2)
    - output layer with one neuron (o1)
    All neurons have same weight/bias
    '''
    def __init__(self):
        # Weights 
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()


    def feedforward(self, x):
        # x is numpy array with two elements
        h1 = sigmoid_function(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid_function(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid_function(self.w5 * h1 + self.w6 * h2 + self.b3)
        
        return o1

    def train(self, data, all_y_trues):
        '''
        data: (n x 2 array), where n = # samples in dataset
        all_y_trues: (n array), where n = # elements
        '''
        learn_rate = 0.1
        # Number of times to loop through dataset
        epochs = 1000 

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Feedforward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid_function(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid_function(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid_function(sum_o1)
                y_pred = o1

                # Calculate partial derivatives

                derivative_y_pred = -2 * (y_true - y_pred)

                # Neuron o1
                derivative_y_pred_w5 = h1 * derive_sigmoid(sum_o1)
                derivative_y_pred_w6 = h2 * derive_sigmoid(sum_o1)
                derivative_y_pred_b3 = derive_sigmoid(sum_o1)

                derivative_y_pred_h1 = self.w5 * derive_sigmoid(sum_o1)
                derivative_y_pred_h2 = self.w6 * derive_sigmoid(sum_o1)

                # Neuron h1
                derivative_h1_w1 = x[0] * derive_sigmoid(sum_h1)
                derivative_h1_w2 = x[1] * derive_sigmoid(sum_h1)
                derivative_h1_b1 = derive_sigmoid(sum_h1)

                # Neuron h2
                derivative_h2_w3 = x[0] * derive_sigmoid(sum_h2)
                derivative_h2_w4 = x[1] * derive_sigmoid(sum_h2)
                derivative_h2_b2 = derive_sigmoid(sum_h2)

                # Update weights/bias
                # h1
                self.w1 -= learn_rate * derivative_y_pred * derivative_y_pred_h1 * derivative_h1_w1
                self.w2 -= learn_rate * derivative_y_pred * derivative_y_pred_h1 * derivative_h1_w2
                self.b1 -= learn_rate * derivative_y_pred * derivative_y_pred_h1 * derivative_h1_b1

                # h2
                self.w3 -= learn_rate * derivative_y_pred * derivative_y_pred_h2 * derivative_h2_w3
                self.w4 -= learn_rate * derivative_y_pred * derivative_y_pred_h2 * derivative_h2_w4
                self.b2 -= learn_rate * derivative_y_pred * derivative_y_pred_h2 * derivative_h2_b2

                # o1
                self.w5 -= learn_rate * derivative_y_pred * derivative_y_pred_w5
                self.w5 -= learn_rate * derivative_y_pred * derivative_y_pred_w6
                self.w5 -= learn_rate * derivative_y_pred * derivative_y_pred_b3

                # Calculate total loss at end of each epoch/cycle
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    # print("Epoch %d loss: %.3f" % (epoch, loss))

network = NeuralNetwork()

# Dataset (4 users)
# Where user is...
# 1. Weight from baseline 135lbs
# 2. Height from baseline 66inches
# 3. Gender (variable for prediction)
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
all_y_trues = np.array([1, 0, 0, 1,])

# Train network :)
network.train(data, all_y_trues)

# Predictions
new_user = np.array([-7, -3]) # 128lbs, 63inches
another_user = np.array([20, 2]) # 1155lbs, 68inches

print('New user: %.3f' % network.feedforward(new_user)) # F (Values closer to 1)
print('Other new user: %.3f' % network.feedforward(another_user)) # M (Values closer to 0)


