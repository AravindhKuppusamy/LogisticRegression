"""
File: logreg.py
Language: Python 3.5.2
Author: Aravindh Kuppusamy ( axk8776@rit.edu )
Description: Logistic Regressor
"""

from math import e
from random import uniform
import matplotlib.pyplot as plt

__author__ = "Aravindh Kuppusamy"


def loadData(file):
    X = list()
    Y = list()
    with open(file) as f:
        for line in f:
            sample = line.strip().split(",")
            X += [[1.0] + [float(sample[0])] + [float(sample[1])]]
            Y += [int(sample[2])]

    return X, Y


def saveWeights(weightList):
    weights = open('weights.csv', 'w')
    for weight in weightList:
        weights.write("%s,%s,%s\n" % (weight[0], weight[1], weight[2]))


def getFinalWeight(weightsFile):
    with open(weightsFile) as f:
        for line in f:
            pass
        W = line.strip().split(',')
        W = [float(wt) for wt in W]
        return W


def sigmoid(x):
    return 1 / (1 + e ** (-x))


def prediction(sample, weights):
    z = [w * i for w, i in zip(weights, sample)]
    return sigmoid(sum(z))


def sqError(x, y):
    return (x-y)**2


def logReg(X, Y, epochs, alpha):

    SSEs = list()
    weights = [uniform(0, 1) for _ in range(3)]
    weightList = [weights]

    for epoch in range(epochs):

        newWeights = weights
        SSE = 0

        for i in range(len(X)):

            h = prediction(X[i], weights)
            delta = (Y[i] - h) * h * (1-h)
            error = [delta * x for x in X[i]]
            newWeights = [w + alpha*err for w, err in zip(newWeights, error)]
            SSE += sqError(Y[i], h)

        weights = newWeights
        weightList += [weights]
        avgSSE = SSE/len(X)
        # print(avgSSE)
        SSEs.append(avgSSE)

    saveWeights(weightList)
    return SSEs


def SSEvsEpoch(fig, SSEs):
    left = fig.add_subplot(121)
    left.set_title("SSE vs Epoch")
    left.set_xlabel("No. of Epochs")
    left.set_ylabel("Sum of Squared Error")
    left.plot(SSEs, label='SSE')
    left.legend(loc="upper right")


def decisionBoundary(fig, X, Y, weights):

    W = getFinalWeight(weights)
    pos = list(), list()
    neg = list(), list()
    for i in range(len(X)):
        if Y[i] == 1:
            pos[0].append(X[i][1])
            pos[1].append(X[i][2])
        if Y[i] == 0:
            neg[0].append(X[i][1])
            neg[1].append(X[i][2])

    right = fig.add_subplot(122)
    right.set_title("Decision Boundary")
    right.set_xlabel("1st Attribute of Input samples")
    right.set_ylabel("2st Attribute of Input samples")
    right.scatter(pos[0], pos[1], marker='o', c='m', label='Class 0')
    right.scatter(neg[0], neg[1], marker='o', c='c', label='Class 1')

    _, mmx1, mmx2 = [(min(a), max(a)) for a in zip(*X)]
    p1 = min(mmx1[0], mmx2[0]), max(mmx1[1], mmx2[1])
    p2 = (-W[0] - W[1]*p1[0])/W[2], (-W[0] - W[1]*p1[1])/W[2]
    right.plot([p1[0], p1[1]], [p2[0], p2[1]], label='Decision Boundary')
    right.legend(loc="best")


def classificationReport(X, Y, weights):

    correct = 0
    w = getFinalWeight(weights)
    for (x, y) in zip(X, Y):
        if prediction(x, w) <= 0.5:
            if y == 0: correct += 1
        else:
            if y == 1: correct += 1
    print("No. of Correctly classified samples      : %d" % correct)
    print("No. of Incorrectly classified samples    : %d" % (len(X) - correct))


def main():
    file = input("Please enter the file name: (ex. nls.csv)")
    epochs = 10000
    learningRate = 0.001
    varValues, target = loadData(file)
    SS = logReg(varValues, target, epochs, learningRate)

    fig = plt.figure()
    SSEvsEpoch(fig, SS)
    decisionBoundary(fig, varValues, target, "weights.csv")
    classificationReport(varValues, target, "weights.csv")
    plt.show()


if __name__ == "__main__":
    main()
