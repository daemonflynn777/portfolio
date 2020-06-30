import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose as sdecomp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from itertools import product
import warnings

# ОТРИСОВКА ВРЕМЕННОГО РЯДА
def ShowPlot(data, message):
    fig = plt.figure(figsize = (19, 8), num = message)
    d = plt.plot(data[:], color = 'purple', label = "data") # ОТРИСОВАВ ТОЛЬКО ПЕРВЫЕ 200 ЗНАЧЕНИЙ, МОЖНО УВИЕТЬ НАЛИЧИЕ СЕЗОНН0СТИ; ПРИМЕРНЫЙ РАЗМЕР СЕЗОНА - 75, начниая с 0-ГО ЗНАЧЕНИЯ
    #p2 = plt.bar(data, height = 1.0, width = 0.2, align = 'edge', color = 'purple')
    plt.legend()
    plt.grid(linewidth = 1)
    plt.title(message, fontsize = 'xx-large')
    plt.show()
    fig = sdecomp(data, model = "additive", freq = 75).plot()
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
    data = data.astype('float')
    data = data.to_numpy()
    data = np.diff(data)
    #ShowPlot(data, "Исходный ряд после дифференцирования")
    print(data.shape)
    warnings.simplefilter('ignore')
    data_len = len(data)
    split = int(data_len*0.8) # РАЗБИВАЕМ РЯД НА ОБУЧАЮЩУЮ И ТЕСТИРУЮУЩУЮ ВЫБОРКИ
    data_train = data[ : split]
    data_test = data[split : ]
    adf_stat, adf_crit_val = adfuller(data, regression = 'c')[0], adfuller(data, regression = 'c')[4]["5%"]
    int_degree = 1
    while adf_stat >= adf_crit_val:
        print("\n", adf_stat, adf_crit_val)
        data = np.diff(data)
        adf_stat, adf_crit_val = adfuller(data, regression = 'c')[0], adfuller(data, regression = 'c')[4]["5%"]
        int_degree += 1
    print("Порядок интегрированности временного ряда равен:", int_degree)
    # НАЙДЕМ ПО ГРАФИКУ КОЭФФИЦИЕНТЫ p И q ДЛЯ МОДЕЛИ ARIMA
    fig = plt.figure(figsize = (8, 8), num = 'Графики ACF и PACF')
    ax = []
    ax.append(fig.add_subplot(2, 1, 1))
    plot_acf(data, ax = ax[0])
    ax.append(fig.add_subplot(2, 1, 2))
    plot_pacf(data, ax = ax[1])
    plt.show()
    print("В ходе тестирования выяснил, что параметры ACF и PACF 2 и 0 соотвественно")
    acf_coef = int(input("Введите коэффициент ACF: ")) # MA составляющая временного ряда, параметр "q"
    pacf_coef = int(input("Введите коэффициент PACF: ")) # AR составляющая временного ряда, параметр "p"
    parameters = product(range(pacf_coef + 1), range(acf_coef + 1))
    parameters_list = list(parameters)
    model_score = model_score_best = q_best = p_best = 0 # БУДЕМ ИСПОЛЬЗОВАТЬ МЕТРИКУ R^2 (можно и MSE), ТАК КАК МЕТРИКА АКАИКЕ ПРИ ТАКИХ p И q НЕ ОБЯЗАТЕЛЬНА
    forecasted_best = list(np.zeros(len(data_test)))
    for params in parameters_list:
        print("\nТестируется ARIMA (%d,%d,%d)" % (params[0], int_degree, params[1]))
        forecasted = []
        for iter in range(len(data_test)):
            model = ARIMA(data_train, order = (params[0], int_degree, params[1]))
            model_fit = model.fit(disp = 0)
            forecast_step = model_fit.forecast()[0]
            forecasted.append(forecast_step)
            data_train = np.append(data_train, data_test[iter])
        #print(len(forecasted))
        #forecasted = np.array(forecasted)
        model_score = r2_score(data_test, forecasted)
        print("Её метрика:", model_score)
        if model_score > model_score_best:
            model_score_best = model_score
            p_best = params[0]
            q_best = params[1]
            forecasted_best = forecasted
    print("Наилучшая модель по метрике R_2 это ARIMA(%d,%d,%d) с результатом %f" % (p_best, int_degree, q_best, model_score_best))
    fig = plt.figure(figsize = (8, 5), num = 'Прогоноз VS Тест')
    plt.plot(data_test, color = 'orange', label = 'Тест')
    plt.plot(forecasted_best, color = 'purple', label = 'Прогноз')
    plt.legend()
    plt.grid(linewidth = 1)
    plt.title('Прогоноз VS Тест', fontsize = 'xx-large')
    plt.show()
    model = ARIMA(data, order = (p_best, int_degree, q_best))
    model_fit = model.fit(disp = 0)
    forecast = []
    for i in [1, 3, 10]:
        forecast.append(model_fit.forecast(i)[0])
    for i, days in enumerate([1, 3, 10]):
        print(f"Прогноз с помощью ARIMA на {days}", forecast[i])
    return forecast

# ЗДЕСЬ РЕАЛИЗОВАН МОЙ СОБСТВЕННЫЙ АЛГОРИТМ ПРОГНОЗОВ
def MyPrediction(data, forecast_range):
    data = data.to_numpy()
    period_lenght = 75 # ДЛИНА СЕЗОНА
    pos = data.shape[0] % period_lenght # ЗДЕСЬ МЫ ПОЛУЧАЕМ ПОЗИЦИЮ ВНУТРИ ПЕРИОДА, НА КОТОРУЮ НУЖНО ДЕЛАТЬ ПРОГНОЗ
    forecast = []
    a = 0.85 # КОЭФФИЦИЕНТ В ЭКСПОНЕНЦИАЛЬНОМ СГЛАЖИВАНИИ
    for fr in range(forecast_range):
        prev = data[pos + fr : : period_lenght] # ЗДЕСЬ МЫ СОСТАВЛЯЕМ СПИСОК ИЗ ПРЕДЫДУЩИХ ЗНАЧЕНИЙ НА ПОЗИЦИИ pos ВНУТРИ ПЕРИОДА
        prev = np.array(prev)
        tmp_forecast = [prev[0]]
        for i in range(len(prev)):
            tmp_forecast.append((a*prev[i] + (1 - a)*prev[i - 1]))
        forecast.append(a*tmp_forecast[len(tmp_forecast) - 1] + (1- a)*tmp_forecast[len(tmp_forecast) - 2])
    return forecast

# ЗДЕСЬ РЕАЛИЗОВАН ПРОГНОЗ С ПОМОЩЬЮ АЛГОРИТМА XBGoost Regressor


# M A I N
#df = GetData("task1.txt")
df = pd.read_csv("task1.csv", index_col = 0)
ShowPlot(df['data'], "Исходные данные")
print(df.describe())
#profile = ProfileReport(df, title="Data")
#profile.to_file("task1.html")
# ПОСМОТРЕТЬ ОТЧЕТ МОЖНО ОТКРЫВ ДАННЫЙ .html ФАЙЛ
ArimaPrediction(df['data'])
for i in [1, 3, 10]:
    print(f"Прогноз собственным методом на {i} шагов вперед:", MyPrediction(df['data'], i))
