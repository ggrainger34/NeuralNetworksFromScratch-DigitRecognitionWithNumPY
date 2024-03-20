import numpy as np

class Layer:
    def __init__(self, prevLayer, activations, weights, neuronCount):
        self.weights = weights
        self.neuronCount = neuronCount
        self.prevLayer = prevLayer
        self.activations = activations

    #Forward pass calculates
    def forward(self):
        if self.prevLayer == None:
            return self.activations
        return self.activationFunction(self.weights, self.prevLayer.activations)

    #Activation Function takes in the weights between the previous layer and the current layer as a matrix
    #And multiplies it with the activations of the previous layer
    def activationFunction(self, weights, prevLayerActivations):
        return weights.dot(prevLayerActivations)

def main():
    a0 = Layer(None, np.array([0.3, 0.2, 0.4]), None, 3)
    a1 = Layer(a0, None, np.array([[1,1,1], [1,1,1]]), 2)

    print(a0.forward())
    print(a1.forward())

if __name__ == '__main__':
    main()