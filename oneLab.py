from tabulate import tabulate
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def normalData(n, a, b):
    result = np.random.random(n)
    for i in range(n):
        result[i] = a + (b-a) * result[i]
    return result


def uniformData(n, a, b):
    return np.random.uniform(a, b, size=n)


def printTable(q, delta, array, result, fileName, n):
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
                    str(np.around(array[i] * n, decimals = 2)),
                    str(np.around(q[i] - n * array[i], decimals=2)),
                    str(np.around(result[i], decimals=2))])

    rows.append([str(len(q)),
                "-",
                str(np.sum(q)),
                str(np.around(np.sum(array), decimals=4)),
                str(np.around(np.sum(array * n), decimals=2)),
                str(-np.around(np.sum(q - n * array), decimals=2)),
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


def countHiSquare(dist, k, table, fileName, start, finish, n):
    delta = np.linspace(start, finish, num=k)
    array = np.array([stats.norm.cdf(delta[0])])
    q = np.array([len(dist[dist <= delta[0]])])
    for i in range(0, len(delta) - 1):
        new_ar = stats.norm.cdf(delta[i + 1]) - stats.norm.cdf(delta[i])
        array = np.append(array, new_ar)
        q = np.append(q, len(dist[(dist <= delta[i + 1]) & (dist >= delta[i])]))
    array = np.append(array, 1 - stats.norm.cdf(delta[-1]))
    q = np.append(q, len(dist[dist >= delta[-1]]))
    result = np.divide(np.multiply((q - n * array), (q - n * array)), array * n)    
    if table == True:
        printTable(q, delta, array, result, fileName, n)
    return (np.around(np.sum(result), decimals=2))


def drawData(data1, data2, k, name1, name2, value, hiNorm, hiUniform):
    interval1 = 1 + math.floor(math.log2(len(data1)))
    interval2 = 1 + math.floor(math.log2(len(data2)))
    f, ax = plt.subplots(1, 2)
    strNorm = str(name1) + ":\n" + str(round(hiNorm,2))
    strUniform = str(name2) + ":\n" + str(round(hiUniform,2))
    
    if hiNorm > value:
        strNorm += " > " + str(round(value,2))
    elif hiNorm < value:
        strNorm += " < " + str(round(value,2))
    else: strNorm += " = " + str(round(value,2))

    if hiUniform > value:
        strUniform += " > " + str(round(value,2))
    elif hiUniform < value:
        strUniform += " < " + str(round(value,2))
    else: strUniform += " = " + str(round(value,2))

    ax[0].hist(data1, bins=interval1)
    ax[1].hist(data2, bins=interval2)

    ax[0].set_title(strNorm)
    ax[1].set_title(strUniform)

    f.tight_layout()
    plt.show()


############################################################################
alpha = 0.05
p = 1 - alpha
k = 6

start = -2
finish = 2

num = 150

print("--------------------------------------------------")

distNorm = normalData(num, start, finish)
distUniform = uniformData(num, start, finish)

value = stats.chi2.ppf(p, k-3)
hiNorm = countHiSquare(distNorm, k-3, True,'dataNorm.csv', start, finish, num)
hiUniform = countHiSquare(distUniform, k-3, True, 'dataUniform.csv', start, finish, num)
result = str(num) + ": quantile = " + str(value) + " hiSquareNorm = " + str(hiNorm) + " hiSquareUniform = " + str(hiUniform)
print(result)
drawData(distNorm, distUniform, k, "Norm", "Uniform", value, hiNorm, hiUniform)