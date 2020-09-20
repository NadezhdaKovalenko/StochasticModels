import random
import numpy as np
import matplotlib.pyplot as plt


def createUniformData(n, start, finish):
    result = []
    for i in range(n):
        result.append(random.uniform(start, finish))
    return result

def hiSquare(n, start, finish):
    uniformData = createUniformData(n, start, finish)
    drawData(uniformData)

    summ = 0.0

    for i in range(n):
        



def drawData(data):
    plt.hist(data, bins = len(data))
    plt.show()





hiSquare(10, 0.0, 10.0)