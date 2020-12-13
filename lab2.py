import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.lines as lines

def readData(fname:str, sizeOneSignal:int):
    """Чтение данных из файла и их преобразование.
    @param fname Название входного файла.
    @param sizeOneSignal Количество элементов
     в одном сигнале."""
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
    data = np.reshape(newData,
    (newData.shape[1]//sizeOneSignal, sizeOneSignal))
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
    """Вычисление критерия Фишера.
    @param signal Выборка, для которой хотим его
     вычислить.
    @param k число интервалов разделения данных."""
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
    plt.plot(ind, signal)
    plt.show()

def plotIntervals(signal, indArray, start, finish):
    """Изображение сигнала с разделением 
    на интервалы однородности.
    @param signal Весь сигнал.
    @param indArray Номер значения сигнала.
    @param start Начала областей.
    @param finish Конец областей."""
    maximum = max(signal)
    minimum = min(signal)
    plt.plot(indArray, signal)

    for i in range(len(start)):
        x1 = [start[i], start[i]]
        y1 = [minimum, maximum]
        plt.plot(x1, y1, 'k--')

    x2 = [finish[len(finish)-1], finish[len(finish)-1]]
    y2 = [minimum, maximum]
    plt.plot(x2, y2, 'k--')
    plt.show()

def createIndArray(array):
    """Создание массива индексов для 
    данных одного сигнала.
    @param array Данные."""
    indAdday = [obj for obj in range(len(array))]
    return indAdday

def deleteIntervals(signal, indSignal):
    """Деление сигнала на области однородности.
    @param signal Сигнал.
    @param indSignal Номера индексов элементов сигнала."""
    start = 0
    finish = 14
    listStart = []
    listFinish = []
    
    listStart.append(start)
    while finish < len(signal):
        fisher = float(0.0)
        while fisher < float(1.0):
            finish += 2
            if finish > len(signal):
                break
            k = (finish - start) // 2
            fisher = applayFisher(signal[start:finish], k)

        if finish <= len(signal):
            listFinish.append(finish)
            start = finish
            listStart.append(start)
            finish += 14
        else:
            listFinish.append(len(signal))

    return listStart, listFinish

def findTypeIntervals(signal, indArray, start, finish):
    # 0 - тип не определен 
    # 1 - выбросы - большое f малой длины
    # 2 - переходный процесс - связан с большим f
    # 3 - фон - 
    # 4 - сигнал - 
    typeI = [0] * len(start)
    fishers = []
    lenIntervals = []
    for i in range(len(start)):
        startInterval = start[i]
        finishInterval = finish[i]
        k = (finishInterval - startInterval) // 2
        fisher = applayFisher(signal[startInterval:finishInterval], k)
        fishers.append(fisher)
        lenIntervals.append(finishInterval - startInterval)
    
    return typeI, fishers

def analizeIntervals(signal, indArray, start, finish):
    """
    """
    maximum = max(signal)
    minimum = min(signal)
    
    typeIntervals, fishers = findTypeIntervals(signal,
     indArray, start, finish)

    print(fishers)

    for i in range(len(start)):
        startInterval = start[i]
        finishInterval = finish[i]
        color = 'b'
        plt.plot(indArray[startInterval:finishInterval], 
        signal[startInterval:finishInterval], color,
         label=str(round(fishers[i], 4)))

        x1 = [startInterval,startInterval]
        y1 = [minimum, maximum]
        plt.plot(x1, y1, 'k--')

        x2 = [finishInterval,finishInterval]
        y2 = [minimum, maximum]
        plt.plot(x2, y2, 'k--')
    
    #plt.legend()
    plt.show()
    

if __name__ == "__main__":

    fileData = readData("wave_ampl.txt", 1024)
    indArray = createIndArray(fileData[0])

    idSignal = 432 # здесь меняем номер сигнала от 0 до 799, можно взять любой
    signal = fileData[idSignal]

    plotSignal(signal, indArray)
    start, finish = deleteIntervals(signal, indArray)
    analizeIntervals(signal, indArray, start, finish)