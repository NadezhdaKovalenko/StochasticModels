import numpy as np
import matplotlib.pyplot as plt
import math

def readData(fname:str, sizeOneSignal:int):
    """Чтение данных из файла и их преобразование.
    @param fname Название входного файла.
    @param sizeOneSignal Количество элементов в одном сигнале."""

    data = []
    with open(fname, 'r') as inf:
        for line in inf.readlines():
            removeStrFirst = line.replace("[", "")
            removeStrSecond = removeStrFirst.replace("]", "")
            data.append(removeStrSecond.split(", "))

    floatData = []
    for oneData in data:
        floatData.append([float(x) for x in oneData])

    newData = np.asarray(floatData)
    data = np.reshape(newData, (newData.shape[1]//sizeOneSignal, sizeOneSignal))
    return data

def findInterGroup(signal):
    """Вычисление межгрупповой дисперсии.
    @param signal Данные."""
    
    summ = 0.0
    mean = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean[i] = np.mean(signal[i])
    meanMean = np.mean(mean)

    for i in range(len(mean)):
        summ += (mean[i]-meanMean)**2
    summ /= (signal.shape[0] - 1)
    result = math.sqrt(summ)
    result = result**2
    return signal.shape[1]*result

def findIntarGroup(signal):
    """Вычисление внутригрупповой дисперсии.
    @param signal Данные."""
    result = 0.0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        summ = 0.0
        for j in range(signal.shape[1]):
            summ += (signal[i][j] - mean)**2
        summ /= (signal.shape[0] - 1)
        result += math.sqrt(summ)

    return result / signal.shape[0]

def applayFisher(signal,k):
    """
    @param signal"""

    newSizeY = int(signal.size / k) 
    newSizeX = k
    splitData = np.reshape(signal, (newSizeX, newSizeY))
    interGroup = findInterGroup(splitData)
    intraGroup = findIntarGroup(splitData)
    return interGroup/intraGroup

def plotSignal(signal, ind):
    """Отрисовка данных одного сигнала.
    @param signal Данные одного сигнала(y).
    @param ind Индексы данных(x)."""

    plt.scatter(ind, signal)
    plt.show()

def createIndArray(array):
    """Создание массива индексов для данных одного сигнала.
    @param array Данные."""

    indAdday = [obj for obj in range(len(array))]
    return indAdday


if __name__ == "__main__":

    fileData = readData("wave_ampl.txt", 1024)
    indArray = createIndArray(fileData[0])

    #idSignal = 6
    k = 32
    #plotSignal(fileData[idSignal], indArray)
    #print(applayFisher(fileData[idSignal], k))

    for idSignal in range(len(fileData)):        
        result = applayFisher(fileData[idSignal],k)
        if result < 23:
            print("ind signal = " + str(idSignal))
            print(result)
            plotSignal(fileData[idSignal], indArray)

    print("done")
