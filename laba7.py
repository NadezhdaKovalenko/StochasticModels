from tabulate import tabulate
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import csv


def normalData(n, a, b):
    result = np.random.random(n)
    for i in range(n):
        result[i] = a + (b-a) * result[i]
    return result


def uniformData(n, a, b):
    return np.random.uniform(a, b, size=n)


def printTable(q, delta, array, result, fileName):
    headersCsv = ['i', 'Delta_i','n_i', 'p_i', 'np_i', 'n_i-np_i', 'frac{(n_i-np_i)^2}{np_i}']
    rows = []
    strRows = []
     
    for i in range(0, len(q)):
        if i == 0:
            boarders = ['-infity', np.around(delta[0], decimals=2)]
        elif i == len(q) - 1:
            boarders = [np.around(delta[-1], decimals=2), 'infity']
        else:
            boarders = [np.around(delta[i - 1], decimals=2), np.around(delta[i], decimals=2)]

        rows.append([str(i + 1),
                    str(boarders),
                    str( q[i]),
                    str(np.around(array[i], decimals=4)),
                    str(np.around(array[i] * 100, decimals = 2)),
                    str(np.around(q[i] - 100 * array[i], decimals=2)),
                    str(np.around(result[i], decimals=2))])

    rows.append([str(len(q)),
                "-",
                str(np.sum(q)),
                str(np.around(np.sum(array), decimals=4)),
                str(np.around(np.sum(array * 100), decimals=2)),
                str(-np.around(np.sum(q - 100 * array), decimals=2)),
                str(np.around(np.sum(result), decimals=2))])


    csv_writer(fileName, headersCsv, rows)
    #print(tabulate(rows, headersTex, tablefmt="latex"))
    print("Writing complete")

def csv_writer(path, fieldnames, data):
    """
    Функция для записи в файл csv
    path - путь до файла
    fieldnames - название столбцов
    data - список из списков
    """
    with open(path, "w", newline='') as out_file:
        '''
        out_file - выходные данные в виде объекта
        delimiter - разделитель :|;
        fieldnames - название полей (столбцов)
        '''
        writer = csv.writer(out_file, delimiter = ";")
        #writer.writeheader()
        for row in data:
            writer.writerow(row)

def printParameters(data):
    print("mu = " +
    str(np.around(np.mean(data), decimals=2)) +
    " sigma = " + str(np.around(np.std(data), decimals=2)))


def countHiSquare(dist, k, table, fileName, start, finish):
    delta = np.linspace(start, finish, num=k-1)
    array = np.array([stats.norm.cdf(delta[0])])
    q = np.array([len(dist[dist <= delta[0]])])
    for i in range(0, len(delta) - 1):
        new_ar = stats.norm.cdf(delta[i + 1]) - stats.norm.cdf(delta[i])
        array = np.append(array, new_ar)
        q = np.append(q, len(dist[(dist <= delta[i + 1]) & (dist >= delta[i])]))
    array = np.append(array, 1 - stats.norm.cdf(delta[-1]))
    q = np.append(q, len(dist[dist >= delta[-1]]))
    result = np.divide(np.multiply((q - 100 * array), (q - 100 * array)), array * 100)    
    if table == True:
        printTable(q, delta, array, result, fileName)
    return (np.around(np.sum(result), decimals=2))


def drawData(data):
    plt.hist(data, bins='auto')
    plt.show()


############################################################################
alpha = 0.05
p = 1 - alpha
k = 501

start = -2
finish = 2

num = 75

print("--------------------------------------------------")


distNorm = normalData(num, start, finish)
distUniform = uniformData(num, start, finish)

value = stats.chi2.ppf(p, k-1)
hiNorm = countHiSquare(distNorm, k, True,'dataNorm.csv', start, finish)
hiUniform = countHiSquare(distUniform, k, True, 'dataUniform.csv', start, finish)
result = str(num)+": quantile = "+str(value)+" hiSquareNorm = "+str(hiNorm)+" hiSquareUniform = "+str(hiUniform)
print(result)
drawData(distNorm)
drawData(distUniform)
