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
