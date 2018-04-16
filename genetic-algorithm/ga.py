from random import randint, random
from os import urandom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def fitnessFunc(individual):
    x = int.from_bytes(individual[0][:2], 'big', signed=False)
    y = int.from_bytes(individual[0][2:], 'big', signed=False)
    return (x ** 2) - (y ** 2)


def calculateFitness(population):
    for individual in population:
        fitness = fitnessFunc(individual)
        individual[1] = fitness

    return population


def calcStats(population):
    popSize = len(population)
    xs = {}
    ys = {}
    maxFit = 0
    avgFit = 0

    for individual in population:
        fit = individual[1]

        if fit > maxFit:
            maxFit = fit
        avgFit += fit

        x = int.from_bytes(individual[0][:2], 'big', signed=False)
        if x not in xs:
            xs[x] = 1
        else:
            xs[x] += 1

        y = int.from_bytes(individual[0][2:], 'big', signed=False)
        if y not in ys:
            ys[y] = 1
        else:
            ys[y] += 1

    avgFit /= len(population)

    xPctConvergence = 0
    yPctConvergence = 0

    for x in xs:
        pctConvergence = xs[x] / popSize
        if pctConvergence > xPctConvergence:
            xPctConvergence = pctConvergence

    for y in ys:
        pctConvergence = ys[y] / popSize
        if pctConvergence > yPctConvergence:
            yPctConvergence = pctConvergence

    converged = ((xPctConvergence > .95) and (yPctConvergence > .95))

    return converged, xPctConvergence, yPctConvergence, maxFit, avgFit


def initPop(popSize):
    population = []

    for i in range(popSize):
        individual = [urandom(4), 0]  # , 0]
        population.append(individual)

    return population


def onePointCrossover(individual1, individual2, stringLength):
    # TODO figure out how to swap at the bit level?
    swapIdx = randint(0, stringLength/8 - 1)

    tmp1 = individual1[0][swapIdx:]
    tmp2 = individual2[0][swapIdx:]

    individual1[0] = individual1[0][:swapIdx] + tmp2
    individual2[0] = individual2[0][:swapIdx] + tmp1

    return individual1, individual2


def twoPointCrossover(individual1, individual2, stringLength):
    # TODO figure out how to swap at the bit level?
    swapIdx1 = randint(0, stringLength/8 - 1)
    swapIdx2 = randint(0, stringLength/8 - 1)
    while swapIdx2 == swapIdx1:
        swapIdx2 = randint(0, stringLength - 1)

    if swapIdx1 > swapIdx2:
        bigIdx = swapIdx1 + 1
        lilIdx = swapIdx2
    else:
        bigIdx = swapIdx2 + 1
        lilIdx = swapIdx1

    tmp1 = individual1[0][lilIdx:bigIdx]
    tmp2 = individual2[0][lilIdx:bigIdx]

    individual1[0] = individual1[0][:lilIdx] + tmp2 + individual1[0][bigIdx:]
    individual2[0] = individual2[0][:lilIdx] + tmp1 + individual2[0][bigIdx:]

    return individual1, individual2


def crossoverProcess(population, crossoverRate, crossoverPoints):
    stringLength = 32

    if crossoverPoints == 1:
        crossoverStrategy = onePointCrossover
    else:
        crossoverStrategy = twoPointCrossover

    for i in range(len(population)):
        if random() < crossoverRate:
            secondParentIdx = i
            while secondParentIdx == i:
                secondParentIdx = randint(0, len(population) - 1)
            population[i], population[secondParentIdx] = crossoverStrategy(population[i],
                                                                           population[secondParentIdx],
                                                                           stringLength)
    return population


def mutate(individual, stringLength):
    num = int.from_bytes(individual[0], 'big', signed=False)
    mask = 1 << randint(0, stringLength - 1)
    individual[0] = int.to_bytes(num ^ mask, 4, 'big', signed=False)
    return individual


def mutationProcess(population, mutationRate):
    stringLength = 32

    for i in range(len(population)):
        if random() < (mutationRate * stringLength):
            population[i] = mutate(population[i], stringLength)

    return population


def populationTournament(population, n):
    popSize = len(population)
    newPop = []

    while len(newPop) < popSize:
        individuals = []
        for i in range(n):
            idx = randint(0, popSize - 1)
            individual = population[idx]
            individuals.append(population[idx])

        individuals.sort(key=lambda x: x[1])
        newPop.append(individuals[-1])

    return newPop


def printStats(maxFit, avgFit, xpct, ypct, iter):
    print("Iteration:", iter)
    print("X Pct Convergence:", xpct)
    print("Y Pct Convergence:", ypct)
    print("Max fitness:", maxFit)
    print("Avg fitness:", avgFit)
    print()


def getGenes(pop):
    xlist = []
    ylist = []
    fitness = []

    for individual in pop:
        x = int.from_bytes(individual[0][:2], 'big', signed=False)
        y = int.from_bytes(individual[0][2:], 'big', signed=False)
        xlist.append(x)
        ylist.append(y)
        fitness.append(individual[1])

    return xlist, ylist, fitness


def graphStats(pop, maxs, avgs, cons):
    # xlist, ylist, fitness = getGenes(pop)

    iters = list(range(len(maxs)))

    plt.subplot(1, 1, 1)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    yellow = mpatches.Patch(color='orange', label='Max fitness')
    blue = mpatches.Patch(color='blue', label='Avg fitness')
    plt.legend(handles=[yellow, blue])
    plt.scatter(iters, avgs)
    plt.scatter(iters, maxs)
    plt.pause(.05)


def printPop(pop):
    for ind in pop:
        x = int.from_bytes(ind[0][:2], 'big', signed=False)
        y = int.from_bytes(ind[0][2:], 'big', signed=False)
        print("X:", x)
        print("Y:", y)
        print("Fitness:", ind[1])


def reproductionProcess(population, crossoverPoints, crossoverRate, mutationRate):
    population = crossoverProcess(population, crossoverRate, crossoverPoints)
    population = mutationProcess(population, mutationRate)

    return population


def geneticAlgorithm(population, mutationRate, crossoverRate, crossoverPoints, verbose=False):
    converged = False
    maxs = []
    avgs = []
    cons = []
    iters = 0
    while not converged:
        converged, xpct, ypct, maxFit, avgFit = calcStats(population)
        maxs.append(maxFit)
        avgs.append(avgFit)
        cons.append((ypct + xpct)/2)
        if verbose:
            printStats(maxFit, avgFit, xpct, ypct, iters)
        if not converged:
            population = calculateFitness(population)
            population = populationTournament(population, 2)
            population = reproductionProcess(population, crossoverPoints, crossoverRate, mutationRate)
            iters += 1
    if verbose:
        graphStats(population, maxs, avgs, cons)

    return max(population, key=lambda x: x[1])


def main():
    popSize = 100
    mutationRate = .001
    crossoverRate = .8
    crossoverPoints = 1

    population = initPop(popSize)


    strongPop = []
    for i in range(1000):
        fittest = geneticAlgorithm(population, mutationRate, crossoverRate, crossoverPoints)
        strongPop.append(fittest)

    geneticAlgorithm(population, mutationRate, crossoverRate, crossoverPoints, True)


main()
