import math
import pickle


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def preprocessor(infile, numDocLines):
    infile = open(infile, "r")

    for i in range(numDocLines):
        infile.readline()

    coordMatrix = infile.read()
    coordMatrix = coordMatrix.split("\n")
    coordMatrix.pop(-1)
    coordMatrix.pop(-1)

    for i in range(len(coordMatrix)):
        coordMatrix[i] = coordMatrix[i].split()[1:]
        for j in range(len(coordMatrix[i])):
            coordMatrix[i][j] = float(coordMatrix[i][j])

    distMatrix = []
    for i in range(len(coordMatrix)):
        distMatrix.append([])
        for j in range(len(coordMatrix)):
            distMatrix[i].append(dist(coordMatrix[i], coordMatrix[j]))

    pList = open("swedenPickle.pk1", 'wb')
    pickle.dump(distMatrix, pList)


preprocessor("sw24978.tsp", 7)
