import numpy as np

from matplotlib import pyplot
from keras.datasets import mnist

class Layer:
    def __init__(self, inputCount, neuronCount):
        self.inputs = inputCount
        self.neuronCount = neuronCount
        self.weights = np.random.randn(inputCount, neuronCount)
        self.biases = np.random.randn(neuronCount)
    
    def forward(self, inputs):
        #Use linear algebra to calculate the output (pre-activation function)
        #Y = AX + B
        output = np.dot(inputs, self.weights) + self.biases
        for i in range(0, len(output)):
            #Run each element of the output vector through the activation function (ReLU)
            #Y = activation(AX + B)
            output[i] = max(0, output[i])
        #Assign the output to an attribute in layer
        self.output = output

class Network:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def feedForward(self, input):
        #Need to feed the input into the first layer before we can run a loop
        self.layers[0].forward(input)
        #Run through all the layers and feed the output from the previous layer into the next
        for i in range(1, len(self.layers)):
            prevOutput = self.layers[i-1].output
            self.layers[i].forward(prevOutput)
        #Return the final layer output
        return self.layers[-1].output

def neuronCost(outputActivation, expectedOutput):
    error = outputActivation - expectedOutput
    return error ** 2

def main():
    #Load the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    pyplot.imshow(train_X[0], cmap=pyplot.get_cmap('gray'))

    #input = train_X[0].flatten() / 255.0

    trainingExampleList, expectedOutputList = train_X[0:5], train_y[0:5]

    #Initialise Network
    network = Network()

    #Initialise each layer in the network
    a0 = Layer(784,16)
    a1 = Layer(16,16)
    a2 = Layer(16,10)

    network.addLayer(a0)
    network.addLayer(a1)
    network.addLayer(a2)

    #Go through each array in the mini batch
    for trainingExample, expectedOutput in zip(trainingExampleList, expectedOutputList):
        #Feed the input through the network
        networkOutput = network.feedForward(trainingExample.flatten() / 255.0)
        costValue = neuronCost(networkOutput, expectedOutput)
        print(costValue)

    pyplot.show()

if __name__ == '__main__':
    main()