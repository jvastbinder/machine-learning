from math import exp, sqrt


def adjustThetasWeights(n, error, guesses, thetas, weights):
    weights[0][0][0] += error[0][0] * n[0]
    weights[0][1][0] += error[0][1] * n[1]

    weights[0][0][1] += error[0][0] * n[0]
    weights[0][1][1] += error[0][1] * n[1]

    weights[1][0][0] += error[1][0] * guesses[0][0]
    weights[1][1][0] += error[1][1] * guesses[0][1]

    weights[1][0][1] += error[1][0] * guesses[0][0]
    weights[1][1][1] += error[1][1] * guesses[0][1]

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            thetas[i][j] += error[i][j]

    return thetas, weights


def calcError(n, guesses):
    error = [[0, 0],
             [0, 0]]

    for i in range(-1, 0 - len(guesses) - 1, -1):
        for j in range(len(guesses[i])):
            guess = guesses[i][j]
            if i == -1:
                error[i][j] = guess * (1 - guess) * (n[j + 2] - guess)
            else:
                error[0][0] = guesses[0][0] * (1 - guesses[0][0]) * ((error[-1][0] * .3) + (error[-1][1] * -.2))
                error[0][1] = guesses[0][1] * (1 - guesses[0][1]) * ((error[-1][0] * .5) + (error[-1][1] * -.4))

    return error


def sumLst(lst):
    total = 0

    for layer in lst:
        for node in layer:
            total += node

    return total


def sigmoid(x):
    return 1 / (1 + exp(-x))


def feedForward(n, thetas, weights):
    guesses = [[0, 0],
               [0, 0]]

    for i in range(len(thetas)):  # for each layer in network
        for j in range(len(thetas[i])):  # for each node in layer
            val = thetas[i][j]
            for k in range(len(weights[i][j])):  # for each connection to node
                if i == 0:  # if first layer use input
                    x = n[k]
                else:  # else use previous layer's output
                    x = guesses[i - 1][k]
                val += (x * weights[i][j][k])
            guesses[i][j] = sigmoid(val)

    return guesses


def backProp(n, guesses, thetas, weights):
    error = calcError(n, guesses)

    theta, weights = adjustThetasWeights(n, error, guesses, thetas, weights)

    return error, thetas, weights


def epoch(data, thetas, weights):
    totalError = 0
    allGuesses = []
    for n in data:
        guesses = feedForward(n, thetas, weights)
        errorLst, thetas, weights = backProp(n, guesses, thetas, weights)
        totalError = sumLst(errorLst)
        allGuesses.append(guesses[-1])
    return allGuesses, totalError, thetas, weights


def main():
    counter = 0
    trainingSet = [(1, 1, 0, 1),
                   (1, 0, 1, 0),
                   (0, 1, 1, 0),
                   (0, 0, 0, 1)]
    guesses = [(1, 1, 0, 1),
               (1, 0, 1, 0),
               (0, 1, 1, 0),
               (0, 0, 0, 1)]
    thetas = [[.1, -.1],
              [-.2, .3]]
    weights = [[[.2, .4], [-.3, .3]],
               [[.3, .5], [-.2, -.4]]]

    error = 1000
    threshold = 0.00001

    while sqrt(error ** 2) > threshold:
        guesses, error, thetas, weights = epoch(trainingSet, thetas, weights)
        counter += 1
        if counter % 1000 == 0:
            print("Training Values (y1 y2)")
            for set in trainingSet:
                for num in set[2:]:
                    print(num, end=' ')
                print()
            print("Output Values (y1 y2)")
            for guess in guesses:
                for num in guess:
                    print('%5.3f ' % num, end='')
                print()
            print("Error: %5.6f" % error)
            print()




if __name__ == "__main__":
    # execute only if run as a script
    main()

