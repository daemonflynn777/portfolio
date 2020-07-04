import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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

def KNNClassifier(dframe):
    #print(dframe.columns)
    dframe = dframe.drop(['x_3', 'x_5','x_6', 'x_7', 'x_8', 'x_9', 'x_10'], axis = 1)
    X = dframe[list(dframe.columns)[ : -1]][ : -20].to_numpy()
    y = dframe[list(dframe.columns)[-1 : ]][ : -20].to_numpy().reshape(len(dframe['y']) - 20, )
    X_validate = dframe[list(dframe.columns)[ : -1]][-20 : ].to_numpy()
    y_validate = dframe[list(dframe.columns)[-1 : ]][-20 : ].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 16)
    model = KNeighborsClassifier(n_jobs = -1)
    cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
    model_gs = GridSearchCV(model,
                            {
                                'n_neighbors': [2, 4, 6, 8],
                                'weights': ['uniform', 'distance']
                            },
                            scoring = 'accuracy',
                            n_jobs = -1,
                            cv = cv_gen
                            )
    model_gs.fit(X, y)
    print(model_gs.best_params_)
    print("Accuracy score", model_gs.best_score_)
    #print(X_validate.shape)
    for i in range(X_validate.shape[0]):
        prediction = model_gs.best_estimator_.predict(X_validate[i].reshape(1, X_validate.shape[1]))
        print("Predicted:", prediction)
        print("Real:", y_validate[i])
    return 0

def RFClassifier(dframe):
    dframe = dframe.drop(['x_6', 'x_7', 'x_8', 'x_9', 'x_10'], axis = 1)
    X = dframe[list(dframe.columns)[ : -1]][ : -20].to_numpy()
    y = dframe[list(dframe.columns)[-1 : ]][ : -20].to_numpy().reshape(len(dframe['y']) - 20, )
    y = np.int_(y)
    X_validate = dframe[list(dframe.columns)[ : -1]][-20 : ].to_numpy()
    y_validate = dframe[list(dframe.columns)[-1 : ]][-20 : ].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 16)
    model = RandomForestClassifier(n_jobs = -1, random_state = 0)
    cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
    model_gs = GridSearchCV(model,
                            {
                                'n_estimators': [75, 100, 125],
                                'criterion': ['gini', 'entropy'],
                                'max_features': ['sqrt', 'log2']
                            },
                            scoring = 'accuracy',
                            n_jobs = -1,
                            cv = cv_gen
                            )
    #print(y)
    model_gs.fit(X, y)
    print(model_gs.best_params_)
    print("Accuracy score", model_gs.best_score_)
    for i in range(X_validate.shape[0]):
        prediction = model_gs.best_estimator_.predict(X_validate[i].reshape(1, X_validate.shape[1]))
        print("Predicted:", prediction)
        print("Real:", y_validate[i])
        print("")
    return 0

def SVMClassifier(dframe):
    dframe = dframe.drop(['x_6', 'x_7', 'x_8', 'x_9', 'x_10'], axis = 1)
    X = dframe[list(dframe.columns)[ : -1]][ : -20].to_numpy()
    y = dframe[list(dframe.columns)[-1 : ]][ : -20].to_numpy().reshape(len(dframe['y']) - 20, )
    y = np.int_(y)
    X_validate = dframe[list(dframe.columns)[ : -1]][-20 : ].to_numpy()
    y_validate = dframe[list(dframe.columns)[-1 : ]][-20 : ].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 16)
    model = SVC(class_weight = 'balanced', verbose = 0)
    cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
    model_gs = GridSearchCV(model,
                            {
                                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                'kernel': ['rbf', 'sigmoid'],
                                'gamma': ['scale', 'auto']
                            },
                            scoring = 'accuracy',
                            n_jobs = -1,
                            cv = cv_gen
                            )
    #print(y)
    model_gs.fit(X, y)
    print(model_gs.best_params_)
    print("Accuracy score", model_gs.best_score_)
    for i in range(X_validate.shape[0]):
        prediction = model_gs.best_estimator_.predict(X_validate[i].reshape(1, X_validate.shape[1]))
        print("Predicted:", prediction)
        print("Real:", y_validate[i])
    return 0

def NeuralClassifier(dframe):
    #print(dframe.columns)
    dframe = dframe.drop(['x_7', 'x_9'], axis = 1)
    X = dframe[list(dframe.columns)[ : -1]][ : -20].to_numpy()
    y = dframe[list(dframe.columns)[-1 : ]][ : -20].to_numpy().reshape(len(dframe['y']) - 20, )
    X_validate = dframe[list(dframe.columns)[ : -1]][-20 : ].to_numpy()
    y_validate = dframe[list(dframe.columns)[-1 : ]][-20 : ].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 16)
    model = MLPClassifier(hidden_layer_sizes = (128, 128,), max_iter = 1000, verbose = 0)
    cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
    model_gs = GridSearchCV(model,
                            {
                                'activation': ['relu'],
                                'solver': ['adam'],
                                'batch_size': [128]
                            },
                            scoring = 'accuracy',
                            n_jobs = -1,
                            cv = cv_gen
                            )
    model_gs.fit(X, y)
    print(model_gs.best_params_)
    print("Accuracy score", model_gs.best_score_)
    #print(X_validate.shape)
    for i in range(X_validate.shape[0]):
        prediction = model_gs.best_estimator_.predict(X_validate[i].reshape(1, X_validate.shape[1]))
        print("Predicted:", prediction)
        print("Real:", y_validate[i])
        print("")
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

for name in list(df.columns)[ : -1]:
    df[name] = pd.Series(np.array([10*val for val in df[name].to_numpy()]))

#GetInfo(df)

class_balance = dict.fromkeys([i for i in range(3, 30)], 0)
for val in df['y']:
    class_balance[val] += 1
print("Процент каждого класса в выборке:", [100*class_balance[key]/len(df['y']) for key in class_balance])

#KNNClassifier(df)

NeuralClassifier(df)
