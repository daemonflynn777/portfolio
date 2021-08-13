import pandas as pd
import numpy as np

from pandas_profiling import ProfileReport
from sklearn.neural_network import MLPRegressor

def GetData(FileName):
    input_file = open(FileName, 'r')
    matrix = []
    for line in input_file:
        matrix.append([float(num) for num in line.split()])
    matrix_np = np.array(matrix)
    columns = [f"x_{i}" for i in range(1, 129)]
    dframe = pd.DataFrame(data = matrix_np, columns = columns)
    dframe.to_csv("task3.csv")
    return dframe

# ПОЛУЧЕНИЕ ОТЧЕТА ПО ДАННЫМ В .html ФАЙЛ
def GetInfo(dframe):
    print(dframe.describe())
    profile = ProfileReport(dframe, title = "Task3 Data")
    profile.to_file("task3.html")

# В ЭТОЙ Ф-ИИ ВСЕ НУЛИ ЗАПОЛНЯЮТСЯ ЗНАЧЕНИЯМИ, СПРОГНОЗИРОВАННЫМИ С ПОМОЩЬЮ MLP Regressor
def NN_filling(dframe, group):
    X = dframe[group[ : -1]].to_numpy()
    y = dframe[group[-1 : ]].to_numpy()
    indexes = np.argwhere(y == float(0))
    index_zero = indexes[0][0] #НОМЕР ПЕРВОЙ СТРОКИ, СОДЕРЖАЩЕЙ 0
    X_train, X_predict = X[ : index_zero], X[index_zero :]
    y_train = y[ : index_zero]
    y_train = y_train.reshape(1, y_train.shape[0])[0]
    model = MLPRegressor(hidden_layer_sizes = (128, 128,), max_iter = 1500, verbose = 0)
    model.fit(X_train, y_train)
    predicted = []
    for features in X_predict:
        prediction = model.predict(features.reshape(1, X_predict.shape[1]))
        predicted.append(prediction[0])
    predicted = np.array(predicted)
    result = pd.Series(np.concatenate([y_train, predicted]))
    return result


# M A I N
df_orig = GetData("task3.txt")
#df_orig = pd.read_csv("task3.csv", index_col = 0)
df = df_orig.copy()
GetInfo(df)
group1 = ['x_19', 'x_35', 'x_47', 'x_48', 'x_77', 'x_91']
group2 = ['x_1', 'x_100', 'x_104', 'x_119', 'x_16', 'x_17', 'x_18', 'x_2', 'x_36', 'x_59', 'x_60', 'x_75', 'x_8', 'x_80', 'x_89', 'x_92', 'x_93', 'x_94', 'x_95', 'x_96']
group3 = ['x_10', 'x_105', 'x_107', 'x_11', 'x_110', 'x_115', 'x_116', 'x_118', 'x_12', 'x_122', 'x_30', 'x_32', 'x_39', 'x_42', 'x_53', 'x_6', 'x_63', 'x_64', 'x_71', 'x_79', 'x_84', 'x_97']
group4 = ['x_108', 'x_109', 'x_113', 'x_117', 'x_121', 'x_123', 'x_124', 'x_125', 'x_127', 'x_13', 'x_21', 'x_23', 'x_3', 'x_33', 'x_4', 'x_5', 'x_54', 'x_55', 'x_58', 'x_61', 'x_67', 'x_7',
          'x_73', 'x_82', 'x_83', 'x_86', 'x_87', 'x_88', 'x_9', 'x_90']
group5 = ['x_101', 'x_102', 'x_103', 'x_106', 'x_111', 'x_112', 'x_114', 'x_120', 'x_126', 'x_128', 'x_14', 'x_15', 'x_20', 'x_22', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_31', 'x_34',
          'x_37', 'x_38', 'x_40', 'x_41', 'x_43', 'x_44', 'x_45', 'x_46', 'x_49', 'x_50', 'x_51', 'x_52', 'x_56', 'x_57', 'x_62', 'x_65', 'x_66', 'x_68', 'x_69', 'x_70', 'x_72', 'x_74', 'x_76',
          'x_78', 'x_81', 'x_85', 'x_98', 'x_99']
new_columns = group1 + group2 + group3 + group4 + group5
df = df[new_columns]
'''
df_orig = df_orig[new_columns]
df_orig.to_csv("task3.csv")
'''
#print(df.sort_values(by = ['x_19'])) УБРАВ КОММЕНТИРОВАНИЕ ЭТОЙ СТРОКИ, А ТАКЖЕ АНАЛОГИЧНОЙ В КОНЦЕ ПРОГРАММЫ, МОЖНО УБЕДИТЬСЯ, ЧТО ВСЕ СОРТИРОВКИ СРАБОТАЛИ ВЕРНО И НИЧЕГО НЕ ПЕРЕМЕШАЛОСЬ
for col in group1:
    df[col].replace(0, df[col].mean(), inplace = True)
for col in group5:
    df[col].replace(0, df[col].mean(), inplace = True)

for col in group2:
    tmp_group = group1
    tmp_group.append(col)
    tmp_df = df.sort_values(by = [col], ascending = False)
    tmp_df[col] = NN_filling(tmp_df, tmp_group)
    df = tmp_df.copy()

group21 = group1 + group2
for col in group3:
    tmp_group = group21
    tmp_group.append(col)
    tmp_df = df.sort_values(by = [col], ascending = False)
    tmp_df[col] = NN_filling(tmp_df, tmp_group)
    df = tmp_df.copy()

group321 = group21 + group3
for col in group4:
    tmp_group = group321
    tmp_group.append(col)
    tmp_df = df.sort_values(by = [col], ascending = False)
    tmp_df[col] = NN_filling(tmp_df, tmp_group)
    df = tmp_df.copy()
'''
df.reset_index(inplace = True)
print(df.sort_values(by = ['x_19'])) # СМ. УКАЗАНИЕ ВЫШЕ
df = df[list(df_orig.columns)]
df.sort_index(inplace = True)
df.reset_index(inplace = True)
'''
df.to_csv("task3filled.csv")
#GetInfo(df)
