import random
import math
import sys


def euclidD(point1, point2):
    sum = 0
    for index in range (len(point1)):
        diff = (point1[index] - point2[index])**2
        sum = sum + diff
    euclidDistance = math.sqrt(sum)
    return euclidDistance


def readFileScores(filename):
    datafile = open(filename, "r")
    datadict = {}
    key = 0
    for aline in datafile:
        key = key + 1
        score = int(aline)
        datadict[key] = [score]
    return datadict


def readFile4(filename):
    datafile = open(filename,"r")
    datadict = {}
    answerkey = {}
    key = 0
    lineNum=0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        sW = float(items[0])
        sL = float(items[1])
        pW = float(items[2])
        pL = float(items[3])
        answer = int(items[4])
        datadict[key] = [sW, sL, pW, pL]
        answerkey[key] = answer
    return datadict, answerkey


def readFileSepal(filename):
    datafile = open(filename,"r")
    datadict = {}
    answerkey = {}
    key = 0
    lineNum=0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        sW = float(items[0])
        sL = float(items[1])
        answer = int(items[4])
        datadict[key] = [sW, sL]
        answerkey[key] = answer
    return datadict, answerkey


def readFilePetal(filename):
    datafile = open(filename,"r")
    datadict = {}
    answerkey = {}
    key = 0
    lineNum=0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        sW = float(items[2])
        sL = float(items[3])
        answer = int(items[4])
        datadict[key] = [sW, sL]
        answerkey[key] = answer
    return datadict, answerkey


def readFileLen(filename):
    datafile = open(filename,"r")
    datadict = {}
    answerkey = {}
    key = 0
    lineNum=0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        sW = float(items[0])
        sL = float(items[2])
        answer = int(items[4])
        datadict[key] = [sW, sL]
        answerkey[key] = answer
    return datadict, answerkey


def readFileWid(filename):
    datafile = open(filename,"r")
    datadict = {}
    answerkey = {}
    key = 0
    lineNum=0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        sW = float(items[1])
        sL = float(items[3])
        answer = int(items[4])
        datadict[key] = [sW, sL]
        answerkey[key] = answer
    return datadict, answerkey


def createCentroids(k, datadict):
    centroids = []
    centroidCount = 0
    centroidKeys = []
    while centroidCount < k:
        rkey = random.randint(1,len(datadict))
        if rkey not in centroidKeys:
            centroids.append(datadict[rkey])
            centroidKeys.append(rkey)
            centroidCount = centroidCount + 1
    return centroids


def createClusters(k, centroids, datadict, repeats, debugFlag, graphFlag):
    oldCentroids = []
    while centroids != oldCentroids:
        clusters = []
        for i in range(k):
            clusters.append([])
        if debugFlag:
            print("Initial Clusters = ",clusters)
            print("Data dictionary = ",datadict)
            if debugFlag:
                print()
            print("Centroids = ",centroids)
        for akey in datadict: #per data point
            distances = []
            for clusterIndex in range(k): #per centroid
                dist = euclidD(datadict[akey],centroids[clusterIndex])
                distances.append(dist)
            if debugFlag:
                print("Distances for point ", akey," = [",end="")
                outString=""
                for z in range(len(distances)):
                    outString+=("%3.1f, " % distances[z])
                outString=outString[:-2]+"]"
                print(outString)
            mindist = min(distances)
            index = distances.index(mindist)
            clusters[index].append(akey)
            displayClusters=[]
            for cluster in clusters:
                clust=[]
                for key in cluster:
                    clust.append(datadict[key])
                displayClusters.append(clust)
        if debugFlag:
            print("Clusters (keys) are now = ",clusters)
            #print("Clusters (values) are now = ",displayClusters)
            for i in range(k):
                print("Cluster Display ",i," = ",displayClusters[i])
            junk=input("press enter to continue . . .")
        dimensions = len(datadict[1])
        oldCentroids = centroids
        for clusterIndex in range(k):
            sums = [0]*dimensions
            for akey in clusters[clusterIndex]:
                datapoints = datadict[akey]
                for ind in range(len(datapoints)):
                    sums[ind] = sums[ind] + datapoints[ind]
            if debugFlag:
                print("Summed distances for cluster ",clusterIndex, " is = ",sums)
            for ind in range(len(sums)):
                clusterLen = len(clusters[clusterIndex])
                if clusterLen != 0:
                    sums[ind] = sums[ind]/clusterLen
            centroids[clusterIndex] = sums
        if debugFlag:
            print("The new centroids are ",centroids)
        count=1
        for c in clusters:
            if debugFlag:
                print ("CLUSTER ", count)
            for key in c:
                if debugFlag:
                    print(datadict[key], end="  ")
            if debugFlag:
                print()
            count=count+1
    return clusters


def getMinsMaxes(datadict):
    xmin=sys.maxint
    xmax=sys.minint
    ymin=sys.maxint
    ymax=sys.minint
    for key in datadict:
        if datadict[key][0]<xmin:
            xmin=datadict[key][0]
        if datadict[key][0]>xmax:
            xmin=datadict[key][0]


def clusterScores(k,numPasses):
    dataDict=readFileScores("scores.txt")
    centroids=createCentroids(k,dataDict)
    clusters=createClusters(k,centroids,dataDict,numPasses,True,True)
    print(clusters)


def checkAnswers(clusters, answerkey):
    correct = 0
    total = 0
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            if answerkey[clusters[i][j]] == i:
                correct += 1
            total += 1
    return correct/total



def clusterEQs():
    numTrials = 100
    print("Number of trials per test:", numTrials)
    print()

    dataDict, answerkey=readFile4("longData.txt")
    avgAccuracy=0
    maxAccuracy=0
    for i in range(numTrials):
        centroids=createCentroids(3,dataDict)
        clusters=createClusters(3,centroids,dataDict,100,False,True)
        accuracy=checkAnswers(clusters, answerkey)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
        avgAccuracy+=accuracy
    print("4 Features")
    print("Avg accuracy =" + str(100 * (avgAccuracy/numTrials)) + "%")
    print("Max accuracy =" + str(maxAccuracy*100) + "%")
    print()

    dataDict, answerkey=readFileSepal("longData.txt")
    avgAccuracy=0
    maxAccuracy=0
    for i in range(numTrials):
        centroids=createCentroids(3,dataDict)
        clusters=createClusters(3,centroids,dataDict,100,False,True)
        accuracy=checkAnswers(clusters, answerkey)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
        avgAccuracy+=accuracy
    print("only sepal length and sepal width")
    print("Avg accuracy =" + str(100 * (avgAccuracy/numTrials)) + "%")
    print("Max accuracy =" + str(maxAccuracy*100) + "%")
    print()

    dataDict, answerkey=readFilePetal("longData.txt")
    avgAccuracy=0
    maxAccuracy=0
    for i in range(numTrials):
        centroids=createCentroids(3,dataDict)
        clusters=createClusters(3,centroids,dataDict,100,False,True)
        accuracy=checkAnswers(clusters, answerkey)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
        avgAccuracy+=accuracy
    print("petal length and petal width")
    print("Avg accuracy" + str(100 * (avgAccuracy/numTrials)) + "%")
    print("Max accuracy=" + str(maxAccuracy*100) + "%")
    print()

    dataDict, answerkey=readFileLen("longData.txt")
    avgAccuracy=0
    maxAccuracy=0
    for i in range(numTrials):
        centroids=createCentroids(3,dataDict)
        clusters=createClusters(3,centroids,dataDict,100,False,True)
        accuracy=checkAnswers(clusters, answerkey)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
        avgAccuracy+=accuracy
    print("petal length and sepal length")
    print("Avg accuracy =" + str(100 * (avgAccuracy/numTrials)) + "%")
    print("Max accuracy =" + str(maxAccuracy*100) + "%")
    print()

    dataDict, answerkey=readFileWid("longData.txt")
    avgAccuracy=0
    maxAccuracy=0
    for i in range(numTrials):
        centroids=createCentroids(3,dataDict)
        clusters=createClusters(3,centroids,dataDict,100,False,True)
        accuracy=checkAnswers(clusters, answerkey)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
        avgAccuracy+=accuracy
    print("petal width and sepal width")
    print("Avg accuracy =" + str(100 * (avgAccuracy/numTrials)) + "%")
    print("Max accuracy =" + str(maxAccuracy*100) + "%")

clusterEQs()
