import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def GetData(FileName):
    input_file = open(FileName, 'r')
    matrix = []
    for line in input_file:
        matrix.append([float(num) for num in line.split()])
    matrix_np = np.array(matrix)
    columns = [f"x_{i}" for i in range(1, 11)]
    columns.append("y")
    dframe = pd.DataFrame(data = matrix_np, columns = columns)
    dframe.to_csv("task2.csv")
    return dframe

def Visualisation(dframe):
    fig = plt.figure(figsize = (21, 8), num = 'Корреляция признаков и ответа')
    ax = []
    names = list(dframe.columns)[ : -1]
    for name in names:
        ax.append(fig.add_subplot(2, 5, names.index(name) + 1))
        sns.regplot(x = 'y', y = name, data = dframe, ax = ax[names.index(name)])
    plt.subplots_adjust(wspace = 0.3)
    plt.show()
    fig.savefig("correlation.png")

    fig = plt.figure(figsize = (12 ,9), num = 'Карта корреляции')
    sns.heatmap(data = dframe.corr(), annot = True, vmin = -1, vmax = 1, center = 0)
    plt.show()
    fig.savefig("heatmap.png")

def GetInfo(dframe):
    print(dframe.describe())
    profile = ProfileReport(dframe, title = "Task2 Data")
    profile.to_file("task2.html")

def KNNClassifier():
    model = KNeighborsClassifier(njobs = -1)
    return 0


# M A I N
#df = GetData("task2.txt")
df = pd.read_csv("task2.csv", index_col = 0)

#Visualisation(df)

'''
for name in ['x_3', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10']:
    df[name] = pd.Series(np.array([1/val for val in df[name].to_numpy()]))
    df[name] = pd.Series(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array([[1/val for val in df[name]]]).reshape(len(df[name]), 1)).reshape(1, len(df[name]))[0])
'''

#GetInfo(df)

class_balance = dict.fromkeys([i for i in range(3, 30)], 0)
for val in df['y']:
    class_balance[val] += 1
print("Процент каждого класса в выборке:", [100*class_balance[key]/len(df['y']) for key in class_balance])

X = df[list(df.columns)[ : -1]]
y = df[list(df.columns)[-1 : ]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 16)
