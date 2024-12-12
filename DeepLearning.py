import numpy as np
import random
import math
from typing import List

def normal_cdf(x, mu = 0, sigma = 1):
    return (1+ math.erf( x-mu ) / math.sqrt(2) / sigma) /2

def inverse_normal_cdf(p, mu = 0, sigma = 1, tolerance = 0.00001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p,tolerance=tolerance)
    
    low_z= -10
    hi_z = 10

    while hi_z - low_z > tolerance:
        mid_z = (low_z+hi_z)/2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z

    return mid_z

def sigmoid(x):
    return 1/(1+np.exp(-x))

def random_uniform(*dim):
    if len(dim) == 1:
        return np.array( [random.random() for _ in range(dim[0])] )
    else:
        return np.array([random_uniform(*dim[1:]) for _ in range(dim[0])])

def random_normal(*dims, mean = 0.0, variance = 1.0):
    if len(dims) == 1:
        return np.array([mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])])
    else:
        return np.array([ random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0]) ])
    
def random_tensor(*dims, init = 'normal'):
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims)/sum(dims)
        return random_normal(*dims, variance=variance)
    
def softMax(input):
    maxVal = max(input)
    input = [ i-maxVal for i in input ]
    input = np.exp(input)
    sumOfEle = sum(input)
    input = [ i/sumOfEle for i in input ]

    return np.array(input)

class Layer:

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradient):
        #Returns the Gradient of the cost w.r.t inputs of this layer i.e outputs of the last one
        raise NotImplementedError
    
    def params(self):
        return np.array([])
    
    def grads(self):
        return np.array([])
    

class Sigmoid(Layer):
    
    sigmoids = np.array([])

    def forward(self, input):
        self.sigmoids = sigmoid(input)
        return self.sigmoids
    
    def backward(self, gradient):
        sig = self.sigmoids

        return sig*(1-sig)*gradient
    
class Linear(Layer):

    input_dim = 0
    output_dim = 0
    w = np.array([])
    b = np.array([])
    b_grad = np.array([])
    w_grad = np.array([])
    input = np.array([])
    name = ''

    def __init__(self, input_dim, output_dim, init = 'xavier', name = '') -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        #Name variable is for debugging purposes
        self.name = name

        self.w = random_tensor(output_dim, input_dim, init=init)
        self.b = random_tensor(output_dim, init=init)


    def forward(self, input):
        self.input = input

        return np.array([np.dot(input, self.w[o]) + self.b[o] for o in range(self.output_dim)])

    def backward(self, gradient):
        self.b_grad = gradient

        #if self.name:
            #print(f'{self.name} \n')
        self.w_grad = np.array([ [ self.input[i] * gradient[o] for i in range(self.input_dim) ] for o in range(self.output_dim) ])
        #Gradient of the cost w.r.t inputs of this layer i.e outputs of the last one
        return np.array([ sum(self.w[o][i] * gradient[o] for o in range(self.output_dim)) for i in range(self.input_dim) ])
    
    def params(self):
        return [self.w, self.b]
    
    def grads(self):
        return [self.w_grad, self.b_grad]
    
class Sequential(Layer):

    layers : List[Layer] = []
    def __init__(self, layers) -> None:
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient
    
    def params(self):
        return [param for layer in self.layers for param in layer.params()]
    
    def grads(self):
        return [grad for layer in self.layers for grad in layer.grads()]
    
class TANH(Layer):
    tanhs = np.array([])

    def forward(self, input):
        self.tanhs = np.tanh(input)
        return self.tanhs
    
    def backward(self, gradient):
        return (1-(self.tanhs**2))*gradient
    
class Dropout(Layer):

    train = False
    p = 0
    mask = np.array([])

    def __init__(self, p):
        self.train = True
        self.p = p

    def dropOrNot(self, input):
        r = random.random()
        return 0 if r < self.p else 1

    def forward(self, input):
        if self.train:
            DNVec = np.vectorize(self.dropOrNot)
            self.mask = DNVec(input)
            return self.mask * input
        else:
            return (1-self.p)*input
        
    def backward(self, gradient):
        if self.train:
            return gradient * self.mask
        else:
            raise RuntimeError("Do Not Call the Backward Function when not training!")
    
#Loss is generalized to make it convienient to experiment with multiple loss functions
class Loss:

    def loss(self, predicted, actual):
        raise NotImplementedError
    
    def gradient(self, predicted, actual):
        raise NotImplementedError
    
class SSE(Loss):

    def loss(self, predicted, actual):
        return (predicted - actual)**2
    
    def gradient(self, predicted, actual):
        return 2*(predicted-actual)
    
#Optimizer is generalized to make it convienient to switch b/w multiple optimizers eg Grad Descent and Momentum
class Optimizer:

    def step(self, layer):
        raise NotImplementedError
    
class GradientDescent(Optimizer):

    learning_rate = 0.1
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def step(self, layer : Layer):
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = param - (self.learning_rate*grad)

class Momentum(Optimizer):

    mo = 0.9
    updates = np.array([])
    learning_rate = 0.1

    def __init__(self, learning_rate = 0.1, mo = 0.9) -> None:
        self.learning_rate = learning_rate
        self.mo = mo
        self.updates = np.array([])

    def step(self, layer : Layer):
        
        if not self.updates:
            self.updates = [ 0*grad for grad in layer.grads() ]

        for update, param, grad in zip (self.updates, layer.params(), layer.grads()):

            update[:] = (self.mo*update) + (1-self.mo)*grad
            param[:] = param - self.learning_rate*update 

class SoftMaxCrossEntropy(Loss):

    def loss(self, predicted, actual):
        probs = softMax(predicted)

        likelihoods = np.log(actual + 1e-30)*actual

        return -sum(likelihoods)
    
    def gradient(self, predicted, actual):
        probs = softMax(predicted)
        return probs - actual