import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose as sdecomp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from itertools import product
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from pmdarima.arima import auto_arima
import warnings

# ОТРИСОВКА ВРЕМЕННОГО РЯДА
def ShowPlot(data, message):
    fig = plt.figure(figsize = (19, 8), num = message)
    d = plt.plot(data[:], color = 'purple', label = "data") # ОТРИСОВАВ ТОЛЬКО ПЕРВЫЕ 200 ЗНАЧЕНИЙ, МОЖНО УВИЕТЬ НАЛИЧИЕ СЕЗОНН0СТИ; ПРИМЕРНЫЙ РАЗМЕР СЕЗОНА - 75, начниая с 0-ГО ЗНАЧЕНИЯ
    plt.legend()
    plt.grid(linewidth = 1)
    plt.title(message, fontsize = 'xx-large')
    plt.show()
    '''
    fig = sdecomp(data, model = "additive", freq = 75).plot()
    plt.show()
    '''

# СЧИТАЕМ ДАННЫЕ ИЗ ФАЙЛА В DataFrame
def GetData(FileName):
    input_file = open(FileName, "r")
    input_data = [int(line) for line in input_file]
    input_file.close()
    dframe = pd.DataFrame(input_data, index = range(len(input_data)), columns = ['data'])
    dframe.to_csv("task1.csv")
    return dframe

# НАХОЖДЕНИЕ НАИЛУЧШИХ ПАРАМЕТРОВ ДЛЯ ARIMA
def ArimaBSTParams(data, logfile):
    data = data.astype('float')
    data = data.to_numpy()
    ShowPlot(data, "Исходный временной ряд")
    ShowPlot(data[ : 75], "Сезонная часть ряда")

    tmp_data = data[ : ]
    adf_stat, adf_crit_val = adfuller(tmp_data, regression = 'c')[0], adfuller(tmp_data, regression = 'c')[4]["5%"]
    int_degree = 0
    while adf_stat >= adf_crit_val:
        print("\n", adf_stat, adf_crit_val)
        tmp_data = np.diff(tmp_data)
        adf_stat, adf_crit_val = adfuller(tmp_data, regression = 'c')[0], adfuller(tmp_data, regression = 'c')[4]["5%"]
        int_degree += 1
    print("Порядок интегрированности временного ряда равен:", int_degree)

    tmp_data_seasonal = data[ : 75]
    adf_stat, adf_crit_val = adfuller(tmp_data_seasonal, regression = 'c')[0], adfuller(tmp_data_seasonal, regression = 'c')[4]["5%"]
    int_degree_seasonal = 0
    while adf_stat >= adf_crit_val:
        print("\n", adf_stat, adf_crit_val)
        tmp_data_seasonal = np.diff(tmp_data_seasonal)
        adf_stat, adf_crit_val = adfuller(tmp_data_seasonal, regression = 'c')[0], adfuller(tmp_data_seasonal, regression = 'c')[4]["5%"]
        int_degree += 1
    print("Порядок интегрированности сезонной части временного ряда равен:", int_degree_seasonal)

    fig = plt.figure(figsize = (8, 8), num = 'Графики ACF и PACF')
    ax = []
    ax.append(fig.add_subplot(2, 1, 1))
    plot_acf(tmp_data, ax = ax[0])
    ax.append(fig.add_subplot(2, 1, 2))
    plot_pacf(tmp_data, ax = ax[1])
    plt.show()

    fig = plt.figure(figsize = (8, 8), num = 'Графики ACF и PACF')
    ax = []
    ax.append(fig.add_subplot(2, 1, 1))
    plot_acf(tmp_data_seasonal, ax = ax[0])
    ax.append(fig.add_subplot(2, 1, 2))
    plot_pacf(tmp_data_seasonal, ax = ax[1])
    plt.show()

    arima_params = auto_arima(y = data, start_p = 0, start_q = 0, start_Q = 0, start_P = 0, max_order = 10, m = 75, seasonal = False, out_of_sample_size = 100)
    print(arima_params.summary())

    forecast = []
    for i, days in enumerate([1, 3, 10]):
        prediction = arima_params.predict(days)
        forecast.append(prediction)
        logfile.write(f"\nПрогноз с помощью auto_arima на {days} шагов вперед: ")
        for num in forecast[i]:
            logfile.write(str(num) + " ")
        logfile.write("\n")
        print(f"Прогноз с помощью auto_arima на {days} шагов вперед:", forecast[i])
    return forecast

    '''
    fig = plt.figure(figsize = (16, 8), num = 'Реальные данные + прогноз')
    dates = np.arange(75 + 3)
    plt.plot(dates[ : 75], data[ : 75], color = 'blue', label = 'real')
    plt.plot(dates[75 : ], forecast, color = 'purple', label = 'predicted')
    plt.legend()
    plt.title('Реальные данные + прогноз', fontsize = 'xx-large')
    plt.show()
    '''

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
        print(f"Прогноз с помощью ARIMA на {days} шагов вперед", forecast[i])
    return forecast

# ЗДЕСЬ РЕАЛИЗОВАН МОЙ СОБСТВЕННЫЙ АЛГОРИТМ ПРОГНОЗОВ
def MyPrediction(data, logfile):
    forecast_range = [1, 3, 10]
    data = data.to_numpy()
    period_lenght = 75 # ДЛИНА СЕЗОНА
    pos = data.shape[0] % period_lenght # ЗДЕСЬ МЫ ПОЛУЧАЕМ ПОЗИЦИЮ ВНУТРИ ПЕРИОДА, НА КОТОРУЮ НУЖНО ДЕЛАТЬ ПРОГНОЗ
    forecast = []
    a = 0.85 # КОЭФФИЦИЕНТ В ЭКСПОНЕНЦИАЛЬНОМ СГЛАЖИВАНИИ
    for ranges in forecast_range:
        sub_forecast = []
        for fr in range(ranges):
            prev = data[pos + fr : : period_lenght] # ЗДЕСЬ МЫ СОСТАВЛЯЕМ СПИСОК ИЗ ПРЕДЫДУЩИХ ЗНАЧЕНИЙ НА ПОЗИЦИИ pos ВНУТРИ ПЕРИОДА
            prev = np.array(prev)
            tmp_forecast = [prev[0]]
            for i in range(len(prev)):
                tmp_forecast.append((a*prev[i] + (1 - a)*prev[i - 1]))
            sub_forecast.append(a*tmp_forecast[len(tmp_forecast) - 1] + (1- a)*tmp_forecast[len(tmp_forecast) - 2])
        forecast.append(sub_forecast)
    for i, days in enumerate([1, 3, 10]):
        logfile.write(f"\nПрогноз своим методом на {days} шагов вперед: ")
        for num in forecast[i]:
            logfile.write(str(num) + " ")
        logfile.write("\n")
        print(f"Прогноз своим методом на {days} шагов вперед", forecast[i])
    return forecast

# ЗДЕСЬ РЕАЛИЗОВАН ПРОГНОЗ С ПОМОЩЬЮ АЛГОРИТМА XBGoost Regressor
def XGBPrediction(data, logfile):
    data = data.to_numpy()
    period = 75
    pos = data.shape[0] % period
    forecast = []
    X = []
    y = []
    for i in range(data.shape[0] - period):
        X.append(data[i : i + period])
        y.append(data[i + period])
    X.append(data[data.shape[0] - period : data.shape[0]])
    data_x = np.array(X) # ПОСЛЕДНЯЯ СТРОКА - ПРИЗНАКИ ДЛЯ ПРОГНОЗА
    data_y = np.array(y)
    xgb_model = xgb.XGBRegressor(learning_rate = 0.001, verbosity = 0, nthread = -1, random_state = 0)
    cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
    xgb_gs = GridSearchCV(
             xgb_model,
             {
                'n_estimators': [75],
                'max_depth': [3, 5],
                'booster': ['gbtree', 'gblinear'],
                'gamma': np.linspace(0.00005, 0.0001, num = 5),
                'reg_alpha': np.linspace(0.0001, 0.001, num = 3),
                'reg_lambda': np.linspace(0.0001, 0.001, num = 3)
             },
             scoring = 'r2',
             n_jobs = -1,
             cv = 5
             )
    xgb_gs.fit(data_x[ : -1], data_y)
    print(xgb_gs.best_params_)
    print("r2 score",xgb_gs.best_score_)
    prediction = xgb_gs.best_estimator_.predict(data_x[-1 : ])
    forecast.append(prediction)
    for i, days in enumerate([1]):
        logfile.write(f"\nПрогноз с помощью XGBoost на {days} шагов вперед: ")
        for num in forecast[i]:
            logfile.write(str(num) + " ")
        logfile.write("\n")
        print(f"Прогноз с помощью XGBoost на {days} шагов вперед", forecast[i])
    return forecast

# ЗДЕСЬ РЕАЛИЗОВАН ПРОГНОЗ С ПОМОЩЬЮ БИБЛИОТЕКИ SKLearn
def SKLearnPrediction(data, logfile):
    data = data.to_numpy()
    period = 75
    pos = data.shape[0] % period
    forecast = []
    X = []
    y = []
    for i in range(int(data.shape[0] / period)):
        X.append(data[i*period : i*period + pos])
        y.append(data[i*period + pos])
    X.append(data[(int(data.shape[0] / period))*period : (int(data.shape[0] / period))*period + pos])
    data_x = np.array(X) # ПОСЛЕДНЯЯ СТРОКА - ПРИЗНАКИ ДЛЯ ПРОГНОЗА
    data_y = np.array(y)
    model = Ridge(copy_X = True, random_state = 0)
    cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
    model_gs = GridSearchCV(
             model,
             {
                'alpha' : np.linspace(0.1, 2.0, num = 10),
                'fit_intercept' : [True, False],
                'normalize' : [True, False],
                'tol' : np.linspace(0.00001, 0.0001, num = 5),
                'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparce_cg', 'sag', 'saga']
             },
             scoring = 'r2',
             n_jobs = -1,
             cv = 5
             )
    model_gs.fit(data_x[ : -1], data_y)
    print(model_gs.best_params_)
    print("r2 score:", model_gs.best_score_)
    prediction = model_gs.best_estimator_.predict(data_x[-1 : ])
    #print("Прогноз с помощью SKLearn Ridge:", prediction)
    forecast.append(prediction)
    for i, days in enumerate([1]):
        logfile.write(f"\nПрогноз с помощью SKLearn на {days} шагов вперед: ")
        for num in forecast[i]:
            logfile.write(str(num) + " ")
        logfile.write("\n")
        print(f"Прогноз с помощью SKLearn на {days} шагов вперед", forecast[i])
    return forecast




# M A I N
df = GetData("task1.txt")
#df = pd.read_csv("task1.csv", index_col = 0)
#ShowPlot(df['data'], "Исходные данные")
print(df.describe())
profile = ProfileReport(df, title="Data")
profile.to_file("task1.html")
# ПОСМОТРЕТЬ ОТЧЕТ МОЖНО ОТКРЫВ ДАННЫЙ .html ФАЙЛ

report = open("task1_report.txt", "w")

ArimaBSTParams(df['data'], report)

MyPrediction(df['data'], report)

XGBPrediction(df['data'], report)

SKLearnPrediction(df['data'], report)

report.close()
