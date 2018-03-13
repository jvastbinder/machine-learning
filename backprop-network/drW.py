import math


class node:
    def __init__(self, initialWts, initialBias):
        self.wts = initialWts
        self.bias = initialBias

    def logistic(self, N):
        return 1 / (1 + math.exp(-N))

    def output(self, inputs):
        total = 0
        for i in range(len(self.wts)):
            total += (self.wts[i] * inputs[i])
        return self.logistic(total + self.bias)

    def cost():
        pass


def dumpNet(hiddenLayer, outputLayer, inputs):
    # Input layer to hidden layer
    print()
    for j in range(len(hiddenLayer)):
        for i in range(inputs):
            print("from input node", i, "to hidden layer node", j, "=", hiddenLayer[j].wts[i])
    # Hidden layer biases
    for j in range(len(hiddenLayer)):
        print("hidden layer bias for node", j, "=", hiddenLayer[j].bias)
    # Hidden layer to output layer
    for j in range(len(outputLayer)):
        for i in range(len(hiddenLayer)):
            print("from hidden layer node", i, "to output layer node", j, "=", outputLayer[j].wts[i])
    # Output layer biases
    for j in range(len(outputLayer)):
        print("output layer bias for node", j, "=", outputLayer[j].bias)
    print()


def forward_prop(trainData, hiddenLayer, outputLayer):
    hiddenLayerOutputs = []
    outputLayerOutputs = []
    for i in range(len(hiddenLayer)):
        hiddenLayerOutputs.append(hiddenLayer[i].output(trainData))
    for i in range(len(outputLayer)):
        outputLayerOutputs.append(outputLayer[i].output(hiddenLayerOutputs))
    # print("hidden layer outputs=",hiddenLayerOutputs)
    # print("output layer outputs=",outputLayerOutputs)
    return hiddenLayerOutputs, outputLayerOutputs


def test_Net(testData, hiddenLayer, outputLayer):
    for pattern in testData:
        hiddenLayerOutputs, outputLayerOutputs = forward_prop(pattern[0:2], hiddenLayer, outputLayer)
        index = 0
        for output in pattern[2:]:
            print("Expected", output, "output", outputLayerOutputs[index])
            index += 1


def XOR_BP_main():
    trainData = [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]]
    testData = [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]]

    hiddenLayer = [node([.2, .4], .1), node([-.3, .3], -.1)]
    outputLayer = [node([.3, .5], -.2), node([-.2, -.4], .3)]
    inputs = 2
    lrate = 50

    # dumpNet(hiddenLayer,outputLayer,inputs)
    trainIndex = 0
    epochCount = 0
    for reps in range(40000):
        # Forward propagation
        hiddenLayerOutputs, outputLayerOutputs = forward_prop(trainData[trainIndex][0:2], hiddenLayer, outputLayer)

        # Back Propogation of error
        # Output Layer
        outputLayerError = []
        for i in range(len(outputLayer)):
            outputLayerError.append(outputLayerOutputs[i] * (1 - outputLayerOutputs[i]) * (
                    trainData[trainIndex][2:][i] - outputLayerOutputs[i]))
        # Hidden Layer
        hiddenLayerError = []
        for i in range(len(hiddenLayer)):
            sumError = 0
            for j in range(len(outputLayer)):
                sumError += outputLayerError[j] * outputLayer[j].wts[i]
            hiddenLayerError.append(hiddenLayerOutputs[i] * (1 - hiddenLayerOutputs[i]) * sumError)

        # Output layer biases calc and apply
        for j in range(len(outputLayer)):
            outputLayer[j].bias += lrate * outputLayerError[j]
        for j in range(len(hiddenLayer)):
            hiddenLayer[j].bias += lrate * hiddenLayerError[j]
        # Hidden layer to output layer weights calc and apply
        for j in range(len(outputLayer)):
            for i in range(len(hiddenLayer)):
                outputLayer[j].wts[i] += lrate * outputLayerError[j] * hiddenLayerOutputs[i]
        # Input Layer to Hidden Layer weights calc and apply
        for j in range(len(hiddenLayer)):
            for i in range(inputs):
                hiddenLayer[j].wts[i] += lrate * hiddenLayerError[j] * trainData[trainIndex][i]

        trainIndex += 1
        if trainIndex == 4:
            trainIndex = 0
            epochCount += 1

    test_Net(testData, hiddenLayer, outputLayer)


XOR_BP_main()
