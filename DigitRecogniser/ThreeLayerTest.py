import numpy as np

class Layer:
    def __init__(self, inputCount, neuronCount):
        self.inputs = inputCount
        self.neuronCount = neuronCount
        self.weights = np.random.randn(inputCount, neuronCount)
        self.biases = np.random.randn(neuronCount)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def main():
    input = [0.2, 0.3, 0.4]

    a0 = Layer(3,2)
    a1 = Layer(2,2)
    a2 = Layer(2,2)

    a0.forward(input)
    a1.forward(a0.output)
    a2.forward(a1.output)

    print(a0.output)
    print(a1.output)
    print(a2.output)

if __name__ == '__main__':
    main()