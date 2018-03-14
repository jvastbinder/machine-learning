from math import exp, sqrt
from random import random

trainingSet, numInputs, numOutputs, numHiddenLayers, numPerLayer = (0,0,0,0,0)


def adjustThetas(error, thetas, lrate):
    for i in range(len(thetas)):
        for j in range(len(thetas[i])):
            thetas[i][j] += error[i][j] * lrate
    return thetas


def adjustWeights(n, error, guesses, weights, lrate):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                if i == 0:
                    guess = n[j]
                else:
                    guess = guesses[i-1][j]
                weights[i][j][k] += error[i][k] * guess * lrate

    return weights


def adjustThetasWeights(n, error, guesses, thetas, weights, lrate):
    weights = adjustWeights(n, error, guesses, weights, lrate)
    thetas = adjustThetas(error, thetas, lrate)

    return thetas, weights


def sumOfErrDrv(weights, error, idx1, idx2):
    sumErr = 0
    for j in range(len(weights[idx1][idx2])):
        sumErr += error[idx1 + 1][j] * weights[idx1 + 1][idx2][j]
    return sumErr


def calcError(n, guesses, weights):
    global numInputs
    error = []

    for i in range(-1, 0 - len(guesses) - 1, -1):
        error.insert(0, [])
        for j in range(len(guesses[i])):
            guess = guesses[i][j]
            if i == -1:
                error[i].append(guess * (1 - guess) * (n[j + numInputs] - guess))
            else:
                error[i].append(guess * (1 - guess) * sumOfErrDrv(weights, error, i, j))
    return error


def feedForward(n, thetas, weights):
    global numOutputs, numInputs
    guesses = []

    for i in range(len(thetas)):  # for each layer in network
        guesses.append([])
        for j in range(len(thetas[i])):  # for each node in layer
            val = thetas[i][j]
            if i == 0:
                length = numInputs
            else:
                length = len(thetas[i - 1])
            for k in range(length):  # for each connection to node
                if i == 0:  # if first layer use input
                    x = n[k]
                else:  # else use previous layer's output
                    x = guesses[i - 1][k]
                val += (x * weights[i][k][j])
            guesses[i].append(sigmoid(val))

    return guesses


# How much control should users have over what runs on their devices?
# Should their be separate feature updates


def sumLst(L):
    if type(L) != list:
        return abs(L)
    if not L:
        return 0
    return sumLst(L[0]) + sumLst(L[1:])


def sigmoid(x):
    return 1 / (1 + exp(-x))


def backProp(n, guesses, thetas, weights, numInputs, lrate):
    error = calcError(n, guesses, weights)
    thetas, weights = adjustThetasWeights(n, error, guesses, thetas, weights, lrate)

    return error, thetas, weights


def epoch(data, thetas, weights, numInputs, lrate):
    totalError = 0
    allGuesses = []

    for n in data:
        guesses = feedForward(n, thetas, weights)
        errorLst, thetas, weights = backProp(n, guesses, thetas, weights, numInputs, lrate)
        totalError = sumLst(errorLst)
        allGuesses.append(guesses[-1])

    return allGuesses, totalError, thetas, weights


def initThetas(numOutputs, numHiddenLayers, numPerLayer):
    thetas = []

    for i in range(numHiddenLayers):
        layer = []
        for j in range(numPerLayer):
            layer.append(random() - .5)
        thetas.append(layer)

    layer = []
    for i in range(numOutputs):
        layer.append(random() - .5)
    thetas.append(layer)

    return thetas


def initWeights(numInputs, thetas):
    weights = []

    inputConnections = []
    for i in range(numInputs):
        inputConnections.append([])
        for j in range(len(thetas[0])):
            inputConnections[i].append(random() - .5)
    weights.append(inputConnections)

    for i in range(0, len(thetas) - 1):  # for layer
        weights.append([])
        for j in range(len(thetas[i])):  # for node
            weights[i+1].append([])
            for k in range(len(thetas[i + 1])):
                weights[i+1][j].append(random() - .5)

    return weights


def initNet(numInputs, numOutputs, numHiddenLayers, numPerLayer):
    thetas = initThetas(numOutputs, numHiddenLayers, numPerLayer)
    weights = initWeights(numInputs, thetas)

    return thetas, weights


def readInData(filename):
    inFile = open(filename, 'r')
    numInputs = int(inFile.readline())
    numOutputs = int(inFile.readline())
    numHiddenLayers = int(inFile.readline())
    numPerLayer = int(inFile.readline())
    rawData = inFile.read().split('\n')

    trainingSet = []
    for example in rawData:
        formatted = example.split()
        realEx = []
        for num in formatted:
            realEx.append(int(num))
        if not (realEx == []):
            trainingSet.append(realEx)

    return trainingSet, numInputs, numOutputs, numHiddenLayers, numPerLayer


def printStats(weights, thetas, trainingSet, guesses, error, prevError):
    print('weights')
    print(weights)
    print('thetas')
    print(thetas)
    print("Training Values")
    for set in trainingSet:
        for num in set[numInputs:]:
            print(num, end=' ')
        print()
    print("Output Values")
    for guess in guesses:
        for num in guess:
            print('%5.3f ' % num, end='')
        print()
    print("Error: %5.6f" % error)
    print("Error Decreasing:", (error < prevError))
    print()


def isRight(guess, ans):
    guessMaxIdx = guess.index(max(guess))
    ansMaxIdx = ans.index(max(ans))
    if guessMaxIdx == ansMaxIdx:
        return 1
    return 0


def printTestSetAccuracy(testingSet, weights, thetas):
    global numInputs
    m = len(testingSet)
    correct = 0
    for n in testingSet:
        guess = feedForward(n, thetas, weights)[-1]
        ans = n[numInputs:]
        correct += isRight(guess, ans)
    print("Accuracy on test set =", 100 * (correct/m))



def main():
    counter = 0
    global trainingSet, numInputs, numOutputs, numHiddenLayers, numPerLayer
    trainingSet, numInputs, numOutputs, numHiddenLayers, numPerLayer = readInData('data.txt')
    testingSet = trainingSet[120:]
    trainingSet = trainingSet[:120]
    thetas, weights = initNet(numInputs, numOutputs, numHiddenLayers, numPerLayer)
    print('weights')
    print(weights)
    print('thetas')
    print(thetas)

    lrate = .1
    error = 1000
    threshold = 0.0001
    prevError = 1000

    while sqrt(error ** 2) > threshold:
        guesses, error, thetas, weights = epoch(trainingSet, thetas, weights, numInputs, lrate)
        error = sqrt(error ** 2)
        if counter % 100 == 0:
            printStats(weights, thetas, trainingSet, guesses, error, prevError)
            print("Epochs:", counter)
            prevError = error
        counter += 1
    printStats(weights, thetas, trainingSet, guesses, error, prevError)
    printTestSetAccuracy(testingSet, weights, thetas)


if __name__ == "__main__":
    main()
