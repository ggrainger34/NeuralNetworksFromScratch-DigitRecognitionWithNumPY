import numpy as np
import random

from keras.datasets import mnist
from matplotlib import pyplot

class Network:
    def __init__(self):
        self.layers = []

    #Method to add a layer to the network
    def addLayer(self, layer):
        self.layers.append(layer)

    #Given an input calculate the output of the network
    def feedForward(self, input):
        #Need to feed the input into the first layer before we can run a loop
        self.layers[0].forward(input)
        #Run through all the layers and feed the 
        #output from the previous layer into the next
        for i in range(1, len(self.layers)):
            prevOutput = self.layers[i-1].getOutput()
            self.layers[i].forward(prevOutput)
        
        finalLayer = self.getFinalLayer()
        
        return finalLayer.output
        
    #Return the output of the final layer
    def getOutput(self):
        return self.layers[-1].getOutput()

    #Return the final layer
    def getFinalLayer(self):
        return self.layers[-1]

    #Adjust ALL gradients in ALL layers using gradient descent
    def applyAllGradients(self, learningRate):
        for layer in self.layers:
            layer.applyGradients(learningRate)
            layer.resetGradients()

    #Calculate the weightGradients and biasGradients for a single example
    def backpropergate(self, inputs, expectedOutput):
        #Feedforward so that the outputs are loaded into the network
        self.feedForward(inputs)

        #Loop backward through all layers
        for layerIndex in range(len(self.layers)-1, -1, -1):

            #Get current layer and associated attributes
            currentLayer = self.layers[layerIndex]
            currentLayerWeightedInputs = currentLayer.z
            currentLayerOutputs = currentLayer.getOutput()

            #If we are dealing with the layer before last, set the previous 
            #Layer output to be input neurons
            if layerIndex == 0: 
                prevLayerOutput = inputs
            else: 
                prevLayerOutput = self.layers[layerIndex-1].getOutput()

            #If we are dealing with the final layer, then we have to treat it differently
            if currentLayer.finalLayer:
                costWrtActivation = 2 * (currentLayerOutputs - expectedOutput)
                actWrtWeightedInput = currentLayer.sigmoidDerivative(currentLayerWeightedInputs)
                weightedInputsWrtWeights = prevLayerOutput
                nodeValues = np.multiply(costWrtActivation, actWrtWeightedInput)

            #If we are dealing with a hidden layer
            else:
                actWrtWeightedInput = currentLayer.sigmoidDerivative(currentLayerWeightedInputs)
                weightedInputsWrtWeights = prevLayerOutput
                weightedInputsWrtActivation = self.layers[layerIndex+1].weights
                nodeValues = np.multiply(np.matmul(
                    np.transpose(weightedInputsWrtActivation), nodeValues), actWrtWeightedInput)

            #Update the weights and biases of the gradient
            #print(nodeValues.shape, np.transpose(weightedInputsWrtWeights).shape)
            currentLayer.weightGradients += np.matmul(nodeValues, np.transpose(weightedInputsWrtWeights))
            currentLayer.biasGradients += nodeValues

    def learn(self, learningRate, processedBatchInputs, processedBatchAnswers):
        for networkInput, expectedNetworkOutput in zip(processedBatchInputs, processedBatchAnswers):
            #print(networkInput.shape, expectedNetworkOutput)
            self.backpropergate(networkInput, expectedNetworkOutput)

        self.applyAllGradients(learningRate)

    #Compute the sum of mean squared error
    def error(self, input, expectedOutput):
        estimatedOutput = self.feedForward(input)
        return np.sum((estimatedOutput - expectedOutput) ** 2)
    
    #Save the weights and biases currently in the network in an associated file
    def saveWeightsAndBiases(self):
        with open('WeightsAndBiases.npy', 'wb') as f:
            for layerIndex in range(0, len(self.layers)):
                currentLayer = self.layers[layerIndex]

                np.save(f, currentLayer.weights)
                np.save(f, currentLayer.biases)

    #Load the weights and biases from a file to the network
    def loadWeightsAndBiases(self):
        try:
            with open('WeightsAndBiases.npy', 'rb') as f:
                print("Loading values into the network")

                for layerIndex in range(0, len(self.layers)):
                    currentLayer = self.layers[layerIndex]

                    w = np.load(f)
                    b = np.load(f)

                    currentLayer.setWeights(w)
                    currentLayer.setBiases(b)

        except FileNotFoundError:
            #If file not found then the network will resort to randomised weights 
            #and biases (which should be handled in the init method for a layer)
            return

class Layer:
    def __init__(self, inputCount, neuronCount, finalLayer):
        self.inputCount = inputCount
        self.weights = np.random.randn(neuronCount, inputCount)
        self.biases = np.random.randn(neuronCount, 1)
        self.neuronCount = neuronCount
        self.finalLayer = finalLayer

        #Initalise the weights gradients as a zero matrix for the specified dimensions
        #Of the layer
        self.weightGradients = np.zeros((neuronCount, inputCount))
        #As there is only one bias per neuron, we only need one value per neuron
        self.biasGradients = np.zeros((neuronCount, 1))
    
    def forward(self, inputs):
        #Calculate weighted input
        z = np.matmul(self.weights, inputs) + self.biases
        
        #Attach weightedInput to the layer
        #Round to 2d.p. for faster computation
        self.z = np.round(z, 2)
        
        #Run weightedInput through activation function
        self.output = self.sigmoid(z)
        
        return self.output
    
    #Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Derivative of sigmoid activation function
    def sigmoidDerivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    #Set all gradients in layer to zero
    def resetGradients(self):
        self.weightGradients = np.zeros((self.neuronCount, self.inputCount))
        self.biasGradients = np.zeros((self.neuronCount, 1))
    
    #Gradient Descent for a single layer
    def applyGradients(self, learningRate):
        self.weights -= np.multiply(learningRate, self.weightGradients)
        self.biases -= np.multiply(learningRate, self.biasGradients)

    def setWeights(self, weights):
        self.weights = weights

    def setBiases(self, biases):
        self.biases = biases

    def getOutput(self):
        return self.output

#Takes in the number the pixels are supposed to represent and returns it in output layer format
def convertAnswersToExpectedOutput(trainingExample):
    #Create a list of 10 zeros
    processedTrainingExample = np.zeros((10,1))

    #Set the value we want the network to guess to be one and everything else to be zero 
    processedTrainingExample[trainingExample] = 1

    return processedTrainingExample

#Convert a single batch into input that is usable by the network
def processBatch(batchInputs, batchAnswers):
    processedInputs = []
    processedOutputs = []

    for input, answer in zip(batchInputs, batchAnswers):
        #Shift the image horizontally
        rollFactor = random.randint(-3,3)
        input = np.roll(input, rollFactor, axis=0)

        #Shift the image vertically
        rollFactor = random.randint(-3,3)
        input = np.roll(input, rollFactor, axis=1)

        """
        #Add noise to improve resiliance
        for x in range(len(input)):
            for y in range(len(input)):
                #Theres a 1 in 10 chance a to change a pixel to random value
                #If the pixel is not part of the digit
                if random.randint(0,10) == 1 and input[x][y] == 0:
                    input[x][y] = random.randint(0,255)
        """
                    
        #Flatten the input
        processedInput = np.transpose(np.array([input.flatten() / 255.0]))
        processedInputs.append(processedInput)

        #Convert the answer to a suitable format
        processedOutput = convertAnswersToExpectedOutput(answer)
        processedOutputs.append(processedOutput)

    return processedInputs, processedOutputs

#Convert all batches into output that is usable by the network
def processAllBatches(batchInputs, batchAnswers):
    #All processed inputs aligned with the answers
    allProcessedInputs = []
    allProcessedAnswers = []

    #Loop through all batches in the complete training data
    for batchInputs, batchAnswers in zip(batchInputs, batchAnswers):
        #Perform preprocessing on every input and answer in a batch
        processedBatchInputs, processedBatchAnswers = processBatch(batchInputs, batchAnswers)

        #Add the now processed batch to the list
        allProcessedInputs.append(processedBatchInputs)
        allProcessedAnswers.append(processedBatchAnswers)

    return allProcessedInputs, allProcessedAnswers

def train(network, learningRate, trainX, trainY, testX, testY):

    #Split the training data into seperate batches, size 64
    trainingBatchInputs = np.array_split(trainX, 64)
    trainingBatchAnswers = np.array_split(trainY, 64)

    #All processed inputs aligned with the answers
    allProcessedInputs, allProcessedAnswers = processAllBatches(trainingBatchInputs, trainingBatchAnswers)

    #Process testing inputs and outputs
    allProcessedTestingInputs, allProcessedTestingAnswers = processBatch(testX, testY)

    #Loop through all of the training data 100 times
    for i in range(0,100):
        #Loop through all of the processed batches
        for processedBatchInputs, processedBatchAnswers in zip(allProcessedInputs, allProcessedAnswers):
            #Adjust the weights and biases to improve performance for that batch
            network.learn(learningRate, processedBatchInputs, processedBatchAnswers)

        totalError = 0

        #Compute the error between predicted values and 
        #expected answers on inputs the network has not been trained on
        for testingInput, testingAnswer in zip(allProcessedTestingInputs, allProcessedTestingAnswers):
            #Sum the total error across all examples
            totalError += network.error(testingInput, testingAnswer)

        #Compute the mean error
        meanError = totalError / len(allProcessedTestingAnswers)

        print(i, "Mean Squared Error:", meanError)   

        #After training save all of the tuned weights and biases
        network.saveWeightsAndBiases()     

    #After training save all of the tuned weights and biases
    network.saveWeightsAndBiases()

def main():
    #Turn off scientific notation
    np.set_printoptions(suppress=True)

    #Load training and testing inputs and answers from the mnist dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()

    #Initialise Network
    network = Network()

    #Define the learning rate
    learningRate = 0.005

    #Define Layers
    a1 = Layer(784,10,False)
    a2 = Layer(10,10,False)
    a3 = Layer(10,10,False)
    a4 = Layer(10,10,True)

    #Add layers
    network.addLayer(a1)
    network.addLayer(a2)
    network.addLayer(a3)
    network.addLayer(a4)

    #Load the saved weights and biases into the network
    network.loadWeightsAndBiases()

	#Train the network
    train(network, learningRate, trainX, trainY, testX, testY)

if __name__ == '__main__':
    main()