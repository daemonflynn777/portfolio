import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# ОТРИСОВКА ВРЕМЕННОГО РЯДА
def ShowPlot(data, message):
    fig = plt.figure(figsize = (19, 8), num = message)
    d = plt.plot(data[ : 200], color = 'purple', label = "data") # ОТРИСОВАВ ТОЛЬКО ПЕРВЫЕ 200 ЗНАЧЕНИЙ, МОЖНО УВИЕТЬ НАЛИЧИЕ СЕЗОНН0СТИ; ПРИМЕРНЫЙ РАЗМЕР СЕЗОНА - 75, начниая с 0-ГО ЗНАЧЕНИЯ
    #p2 = plt.bar(data, height = 1.0, width = 0.2, align = 'edge', color = 'purple')
    plt.legend()
    plt.grid(linewidth = 1)
    plt.title(message, fontsize = 'xx-large')
    plt.show()

# СЧИТАЕМ ДАННЫЕ ИЗ ФАЙЛА В DataFrame
def GetData(FileName):
    input_file = open(FileName, "r")
    input_data = [int(line) for line in input_file]
    input_file.close()
    dframe = pd.DataFrame(input_data, index = range(len(input_data)), columns = ['data'])
    #dframe.index = range(len(dframe['data']))
    dframe.to_csv("task1.csv")
    return dframe

# ПРОГНОЗИРОВАНИЕ ВРЕМЕННОГО РЯДА С ПОМОЩЬЮ ARIMA
def ArimaPrediction(data):
    warnings.simplefilter('ignore')
    data_len = len(data)
    split = int(data_len*0.8)
    data_train = data[ : split]
    data_test = data[split : ]
    adf_stat, adf_crit_val = adfuller(data, regression = 'c')[0], adfuller(data, regression = 'c')[4]["5%"]
    int_degree = 0
    while adf_stat >= adf_crit_val:
        print("\n", adf_stat, adf_crit_val)
        data = np.diff(data)
        adf_stat, adf_crit_val = adfuller(data, regression = 'c')[0], adfuller(data, regression = 'c')[4]["5%"]
        int_degree += 1
    print("Порядок интегрированности временного ряда равен:", int_degree)

#df = GetData("task1.txt")
df = pd.read_csv("task1.csv", index_col = 0)
ShowPlot(df['data'], "Исходные данные")
print(df.describe())
#profile = ProfileReport(df, title="Data")
#profile.to_file("task1.html")
# ПОСМОТРЕТЬ ОТЧЕТ МОЖНО ОТКРЫВ ДАННЫЙ .html ФАЙЛ
ArimaPrediction(df['data'])
