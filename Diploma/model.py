import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.linalg as la
import math

class EconModel():
    def __init__(self, dataset):
        self.aggregated_data_df = pd.read_excel(dataset)
        #self.parameters = ["alpha", "beta", "a_X_XL", "a_X_XO", "b_X_XK",
        #                   "mu_K_X"]

        self.aggregated_data_df["ln_Y_X_const_Y_0"] = np.log(self.aggregated_data_df["Y_X_const"] / list(self.aggregated_data_df["Y_X_const"])[0])
        self.aggregated_data_df['dQ_X'] = np.diff(self.aggregated_data_df['Q_X'], append = self.aggregated_data_df['Q_X'][-1 : ])
        #self.aggregated_data_df['Q_L'] = self.aggregated_data_df['employment_rate'] * self.aggregated_data_df['population']
        self.aggregated_data_df['dQ_L'] = np.diff(self.aggregated_data_df['Q_L'], append = self.aggregated_data_df['Q_L'][-1 : ])
        self.aggregated_data_df['W_X'] = (self.aggregated_data_df['W_X_part']/100)*self.aggregated_data_df['Y_X_const']
        self.aggregated_data_df['dW_X'] = np.diff(self.aggregated_data_df['W_X'], append = self.aggregated_data_df['W_X'][-1 : ])
        self.aggregated_data_df['Q_K'] = self.aggregated_data_df['Q_K'] + self.aggregated_data_df['w']
        self.aggregated_data_df['dQ_K'] = np.diff(self.aggregated_data_df['Q_K'], append = self.aggregated_data_df['Q_K'][-1 : ])
        self.aggregated_data_df['Z_X'] = self.aggregated_data_df['Z_X_part'] * self.aggregated_data_df['Y_X']
        self.aggregated_data_df['dZ_X'] = np.diff(self.aggregated_data_df['Z_X'], append = self.aggregated_data_df['Z_X'][-1 : ])
        self.aggregated_data_df['dW_G'] = self.aggregated_data_df['Y_X']*self.aggregated_data_df['W_G']
        self.aggregated_data_df['W_G'] = self.aggregated_data_df['W_G_revenue']
        #self.aggregated_data_df['W_L'] = 
                                        #self.aggregated_data_df['Q_L']*self.aggregated_data_df['s_L_X'] #-\
                                        #self.aggregated_data_df['W_L_spendings']
        self.aggregated_data_df['dW_L'] = np.diff(self.aggregated_data_df['W_L'], append = self.aggregated_data_df['W_L'][-1 : ])
        self.aggregated_data_df['R'] = self.aggregated_data_df['R']*self.aggregated_data_df['w']
        self.aggregated_data_df['dR'] = np.diff(self.aggregated_data_df['R'], append = self.aggregated_data_df['R'][-1 : ])
        self.aggregated_data_df['dp_X_L'] = np.diff(self.aggregated_data_df['p_X_L'], append = self.aggregated_data_df['p_X_L'][-1 : ])
        self.aggregated_data_df['ds_L_X'] = np.diff(self.aggregated_data_df['s_L_X'], append = self.aggregated_data_df['s_L_X'][-1 : ])
        self.aggregated_data_df['w_hat'] = 1 / self.aggregated_data_df['w']

    def CreateDatasets(self):
        '''
        Создадим датасет для идентификации
        параметров производственной функции (ВВП)
        '''
        #ln_Y_0 = list(self.aggregated_data_df["ln_Y_X_const"])[0]
        Q_L0_X = list(self.aggregated_data_df['Q_L'])[0]
        Q_K0_X = list(self.aggregated_data_df['Q_K'])[0]
        self.GDP_Dataset = pd.DataFrame(data = np.zeros((20, 2)), columns = ['alpha', 'target'])
        self.GDP_Dataset['target'] = self.aggregated_data_df['ln_Y_X_const_Y_0'] -\
                                     np.log(self.aggregated_data_df['Q_K'] / list(self.aggregated_data_df['Q_K'])[0])
        self.GDP_Dataset['alpha'] = np.log(self.aggregated_data_df['Q_L'] / list(self.aggregated_data_df['Q_L'])[0]) -\
                                    np.log(self.aggregated_data_df['Q_K'] / list(self.aggregated_data_df['Q_K'])[0])
        self.GDP_Dataset['label'] = "GDP_Dataset"

        '''
        Объединение нормативов (параметров)
        для всех остальных временных рядов
        '''
        self.parameters = ["target", "a_X_XL", "a_X_XO", "b_X_XK", "mu_K_X", "b_L_XL", "a_L_X", "b_X_GX", "b_M_LO", "b_Z_XB", "b_L_GL", "b_X_LX"]
        self.labeles = ["Q_X_Dataset", "Q_K_Dataset", "Q_L_Dataset", "W_X_Dataset", "Z_X_Dataset", "W_G_Dataset", "R_Dataset", "W_L_Dataset"]

        '''
        Создадим датасет для идентификации
        параметров изменения блага производственного сектора
        '''
        self.Q_X_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_X_Dataset['target'] = self.aggregated_data_df['Y_X'] - self.aggregated_data_df['dQ_X']
        self.Q_X_Dataset['a_X_XL'] = self.aggregated_data_df['Q_X']
        self.Q_X_Dataset['a_X_XO'] = self.aggregated_data_df['Q_X']
        self.Q_X_Dataset['b_X_XK'] = self.aggregated_data_df['W_X'] / self.aggregated_data_df['p_X_X']
        for label in  [x for x in self.parameters if x not in ['target', 'a_X_XL', 'a_X_XO', 'b_X_XK']]:
            self.Q_X_Dataset[label] = 0
        self.Q_X_Dataset['label'] = "Q_X_Dataset"
        self.Q_X_Dataset = self.Q_X_Dataset[ : 5]

        '''
        Создадим датасет для идентификации
        параметров изменения запаса капитала (Q_K_X)
        '''
        self.Q_K_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_K_Dataset['target'] = self.aggregated_data_df['dQ_K']
        self.Q_K_Dataset['b_X_XK'] = self.aggregated_data_df['W_X'] / (self.aggregated_data_df['p_X_X'] * self.aggregated_data_df['c_X_X'])
        self.Q_K_Dataset['mu_K_X'] = self.aggregated_data_df['Q_K'] * (-1)
        for label in [x for x in self.parameters if x not in ['target', 'b_X_XK', 'mu_K_X']]:
            self.Q_K_Dataset[label] = 0
        self.Q_K_Dataset['label'] = "Q_K_Dataset"
        self.Q_K_Dataset = self.Q_K_Dataset[ : 5]
        
        '''
        Создадим датасет для идентификации
        параметров изменения запаса труда (Q_L_X)
        '''
        self.Q_L_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_L_Dataset['target'] = self.aggregated_data_df['dQ_L']
        self.Q_L_Dataset['b_L_XL'] = self.aggregated_data_df['W_X'] / self.aggregated_data_df['s_L_X']
        self.Q_L_Dataset['a_L_X'] = self.aggregated_data_df['Q_L'] * (-1)
        for label in [x for x in self.parameters if x not in ['target', 'b_L_XL', 'a_L_X']]:
            self.Q_L_Dataset[label] = 0
        self.Q_L_Dataset['label'] = "Q_L_Dataset"
        self.Q_L_Dataset = self.Q_L_Dataset[ : 5]

        '''
        Создадим датасет для идентификации
        параметров изменения запаса денег
        в производственном секторе (W_X)
        '''
        self.W_X_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        n_1 = self.aggregated_data_df['n_1']
        n_2 = self.aggregated_data_df['n_2']
        n_4 = self.aggregated_data_df['n_4']
        n_5 = self.aggregated_data_df['n_5']
        self.W_X_Dataset['target'] = self.aggregated_data_df['dW_X'] - self.aggregated_data_df['C_BX'] +\
                                     self.aggregated_data_df['T_3']*(1 - n_1 - n_2 + n_1*n_2)
        self.W_X_Dataset['a_X_XL'] = (self.aggregated_data_df['p_X_L']*self.aggregated_data_df['Q_X'])*\
                                        (1 - n_1 - n_2 + n_1*n_2)
        self.W_X_Dataset['a_X_XO'] = self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']*\
                                        (1 - n_1 - n_2 - n_5 - n_1*n_2*n_5 + n_1*n_5 + n_2*n_5 + n_1*n_2)
        self.W_X_Dataset['b_L_XL'] = self.aggregated_data_df['W_X'] * (n_1 + n_2*n_4 + n_1*n_4 - n_4 - n_1*n_2*n_4 - 1)
        self.W_X_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X']*(n_1 + n_2 - n_1*n_2)
        self.W_X_Dataset['b_X_GX'] = self.aggregated_data_df['W_G']
        for label in [x for x in self.parameters if x not in ['target', 'a_X_XL', 'a_X_XO', 'b_L_XL', 'b_X_GX', 'b_Z_XB']]:
            self.W_X_Dataset[label] = 0
        self.W_X_Dataset['label'] = "W_X_Dataset"
        self.W_X_Dataset = self.W_X_Dataset[ : 5]

        '''
        Создадим датасет для идентификации
        параметров изменения задолженности
        производственного сектора (Z_X)
        '''
        self.Z_X_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Z_X_Dataset['target'] = self.aggregated_data_df['dZ_X'] #- self.aggregated_data_df['H_XB']
        # *self.aggregated_data_df['w']
        self.Z_X_Dataset['a_X_XO'] = (1 - self.aggregated_data_df['ksi'])*(1/self.aggregated_data_df['ksi'])*\
                                     self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']
        self.Z_X_Dataset['b_M_LO'] = (self.aggregated_data_df['ksi'] - 1)*(1/self.aggregated_data_df['ksi'])*self.aggregated_data_df['W_L']
                                        #self.aggregated_data_df['w_hat'] # под вопросом
        #self.Z_X_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X'] * (-1)
        for label in [x for x in self.parameters if x not in ['target', 'a_X_XO', 'b_M_LO']]:
            self.Z_X_Dataset[label] = 0
        self.Z_X_Dataset['label'] = "Z_X_Dataset"
        self.Z_X_Dataset = self.Z_X_Dataset[ : 5]

        '''
        Создадим датасет для идентификации
        параметров изменения запаса денег
        у населения (W_L)
        '''
        self.W_L_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.W_L_Dataset['target'] = self.aggregated_data_df['dW_L'] + self.aggregated_data_df['T_7'] + \
                                     self.aggregated_data_df['C_BX']
        #self.W_L_Dataset['a_X_XO'] = (1 - self.aggregated_data_df['ksi'])*(1/self.aggregated_data_df['ksi'])*self.aggregated_data_df['w']*\
        #                             self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']
        self.W_L_Dataset['b_L_XL'] = self.aggregated_data_df['W_X']
        self.W_L_Dataset['b_L_GL'] = self.aggregated_data_df['W_G']
        self.W_L_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X']
        self.W_L_Dataset['b_X_LX'] = self.aggregated_data_df['W_L'] * (-1)
        self.W_L_Dataset['b_M_LO'] = self.aggregated_data_df['W_L'] * (-1 +\
                                   ((self.aggregated_data_df['ksi'] - 1)/self.aggregated_data_df['ksi'])*(self.aggregated_data_df['w']/\
                                                                                                          self.aggregated_data_df['w_hat']))
        for label in [x for x in self.parameters if x not in ['target', 'a_X_XO', 'b_L_XL', 'b_L_GL', 'b_M_LO', 'b_X_LX', 'b_Z_XB']]:
            self.W_L_Dataset[label] = 0
        self.W_L_Dataset['label'] = "W_L_Dataset"
        self.W_L_Dataset = self.W_L_Dataset[ : 5]  

        '''
        Создадим датасет для идентификации
        параметров изменения ставки
        заработной платы (s_L_X)
        '''

        '''
        Создадим датасет для идентификации
        параметров изменения денег в
        консолидированном бюджете
        '''
        self.W_G_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.W_G_Dataset['target'] = self.aggregated_data_df['dW_G'] - self.aggregated_data_df['T_7'] - self.aggregated_data_df['T_3']*(1 - n_1 - n_2 - n_1*n_2)
        n_6 = self.aggregated_data_df['n_6']
        self.W_G_Dataset['b_M_LO'] = n_6*self.aggregated_data_df['W_L']
        self.W_G_Dataset['a_X_XO'] = self.aggregated_data_df['w']*self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']*\
                                     (n_1 + n_2 + n_5 + n_1*n_2*n_5 - n_1*n_2 - n_1*n_5 - n_2*n_5)
        self.W_G_Dataset['b_L_XL'] = self.aggregated_data_df['W_X']*(n_4 - n_1*n_4 + n_1*n_2*n_4 - n_1 - n_2*n_4)
        self.W_G_Dataset['a_X_XL'] = self.aggregated_data_df['p_X_L']*self.aggregated_data_df['Q_X']*\
                                     (n_1 + n_2 - n_1*n_2)
        self.W_G_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X']*(n_1*n_2 - n_1 - n_2)
        self.W_G_Dataset['b_X_GX'] = self.aggregated_data_df['W_G'] * (-1)
        self.W_G_Dataset['b_L_GL'] = self.aggregated_data_df['W_G'] * (-1)
        for label in [x for x in self.parameters if x not in ['target', 'b_M_LO', 'a_X_XO', 'b_L_XL', 'a_X_XL', 'b_Z_XB', 'b_X_GX', 'b_L_GL']]:
            self.W_G_Dataset[label] = 0
        self.W_G_Dataset['label'] = "W_G_Dataset"
        self.W_G_Dataset = self.W_G_Dataset[ : 5]    

        '''
        Создадим датасет для идентификации
        параметров изменения золотовалюиных
        резервов
        '''
        self.R_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.R_Dataset['target'] = self.aggregated_data_df['dR']
        self.R_Dataset['a_X_XO'] = self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']
        self.R_Dataset['b_M_LO'] = self.aggregated_data_df['W_L'] / self.aggregated_data_df['w_hat'] * (-1)
        for label in [x for x in self.parameters if x not in ['target', 'a_X_XO', 'b_M_LO']]:
            self.R_Dataset[label] = 0
        self.R_Dataset['label'] = "R_Dataset"
        self.R_Dataset = self.R_Dataset[ : 5]
        
        self.FullDataset  = pd.concat([self.Q_X_Dataset, self.Q_K_Dataset, self.Q_L_Dataset, self.W_X_Dataset,\
                                       self.Z_X_Dataset, self.W_G_Dataset, self.R_Dataset, self.W_L_Dataset], ignore_index = True)
        self.FullDataset.to_excel("FullData.xlsx")
        self.FullDatasetScaled = self.FullDataset[self.parameters]
        for i in list(self.FullDatasetScaled.index):
            #self.FullDatasetScaled.loc[i, : ] = self.FullDatasetScaled.loc[i, : ]/max(list(self.FullDatasetScaled.loc[i, : ]))
            #self.FullDatasetScaled.loc[i, : ] = self.FullDatasetScaled.loc[i, : ]/self.FullDatasetScaled.loc[i, 'target']
            self.FullDatasetScaled.loc[i, : ] = self.FullDatasetScaled.loc[i, : ]/(10**len(str(abs(int(self.FullDatasetScaled.loc[i, 'target'])))))
            #print(self.FullDataset.loc[4, : ])
        #print(self.FullDataset)
        self.FullDatasetScaled.to_excel("FullDataScaled.xlsx")

        self.y = self.FullDatasetScaled[['target']].to_numpy().reshape(1, -1)[0]
        self.X = self.FullDatasetScaled[self.parameters[1 : ]].to_numpy()
        self.FullDatasetScaled['label'] = self.FullDataset['label']
        #print(self.X.shape)
        #print(len(self.parameters))
        #print(self.FullDatasetScaled[self.parameters])

    def __TargetFunction(self, point):
        '''
        Функция вычисления функционала
        '''
        return np.dot((np.dot(self.X, point)).T, np.dot(self.X, point)) - 2*np.dot((np.dot(self.X, point)).T, self.y) + np.dot(self.y, self.y)

    def __TargetFunctionGradient(self, point):
        '''
        Функция вычилсения градиента функицонала
        '''
        return 2*np.dot(self.X.T, np.dot(self.X, point)) - 2*np.dot(self.X.T, self.y)

    def __PointProjection(self, point):
        '''
        Функция вычисления проекции на допустимое множество
        '''
        return np.clip(point, 0, 1)

    def __Armiho(self, point, alpha_hat = 5.0, teta = 0.8, eps = 0.0001):
        '''
        Правило Армихо для одномерного поиска
        '''
        point_alpha = np.clip(point - alpha_hat*self.__TargetFunctionGradient(point), 0, 1)
        while self.__TargetFunction(point_alpha) > self.__TargetFunction(point) + eps*np.dot(self.__TargetFunctionGradient(point), point_alpha - point):
            alpha_hat *= teta
            point_alpha = np.clip(point - alpha_hat*self.__TargetFunctionGradient(point), 0, 1)
        return alpha_hat


    def Optimize(self, eps = 0.00001, max_iter_no_change = 1000):
        '''
        Воспользуемся методом проекции градиента
        для нахождения параметров, указанных в
        self.parameters
        '''
        TargetFunctionVals = []
        CurrPoint = np.random.standard_normal(len(self.parameters) - 1)
        #print(self.__TargetFunction(CurrPoint))
        alpha = self.__Armiho(CurrPoint)
        NextPoint = np.clip(CurrPoint - alpha*self.__TargetFunctionGradient(CurrPoint), 0, 1)
        TargetFunctionVals.append(self.__TargetFunction(NextPoint))
        #while self.__TargetFunction(CurrPoint) - self.__TargetFunction(NextPoint) > eps:
        while la.norm(NextPoint - np.clip(NextPoint - alpha*self.__TargetFunctionGradient(NextPoint), 0, 1)) >= eps:
            TargetFunctionVals.append(self.__TargetFunction(NextPoint))
            CurrPoint = NextPoint.copy()
            alpha = self.__Armiho(CurrPoint)
            NextPoint = np.clip(CurrPoint - alpha*self.__TargetFunctionGradient(CurrPoint), 0, 1)
        print(NextPoint)
        self.FullDataset['target_predicted'] = np.dot(self.FullDataset[self.parameters[1 : ]], NextPoint)
        self.FullDataset.to_excel("FullData.xlsx")
        fig = plt.figure(figsize = (12, 4), num = "Loss function")
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(TargetFunctionVals)), TargetFunctionVals)
        ax.set_title('Loss function (squared error)')
        ax.set_xlabel("Номер итерации")
        ax.set_ylabel("Квадрат ошибки")
        plt.show()
        self.solution_main = NextPoint.copy()
        SolutionFixed = []
        for elem in self.solution_main:
            if elem == 0.0:
                SolutionFixed.append(0.001)
            elif elem == 1.0:
                SolutionFixed.append(0.999)
            else:
                SolutionFixed.append(elem)
        self.solution_main = np.array(SolutionFixed)
        self.solution = {key: value for key, value in zip(np.array(self.parameters)[1 : ],\
                                                               self.solution_main)}
        print(self.solution)
        
        '''
        Найдем параметры альфа и бета в уравнении
        для ВВП, используя метод наименьших квадратов
        '''
        self.alpha = sum(self.GDP_Dataset['target']*self.GDP_Dataset['alpha'])/\
                     sum(self.GDP_Dataset['alpha']*self.GDP_Dataset['alpha'])
        if (self.alpha > 1):
            self.alpha = 1.0 - 0.01
        elif (self.alpha) < 0:
            self.alpha = 0.0 + 0.01
        self.beta = 1 - self.alpha

        '''
        Найдем параметры alpha_L_X и alpha_X_L
        для индекса цен и ставки заработной платы
        '''
        self.alpha_X_L_Dataset = pd.DataFrame(data = np.zeros((20, 2)), columns = ['alpha_X_L', 'target'])
        self.alpha_X_L_Dataset['target'] = self.aggregated_data_df['dp_X_L']
        self.alpha_X_L_Dataset['alpha_X_L'] = self.solution['b_X_LX']*self.aggregated_data_df['W_L']


    def __Theil(self, y_true, y_pred):
        return math.sqrt(np.dot(y_true - y_pred, y_true - y_pred).reshape(1, -1)[0].mean())/\
               (math.sqrt(np.dot(y_true, y_true).reshape(1, -1)[0].mean()) +\
                math.sqrt(np.dot(y_pred, y_pred).reshape(1, -1)[0].mean()))

    def BackTest(self, target_type = ""):
        '''
        Посмотрим на коэффициент детерминации
        для прогноза каждого ряда в отдельности
        '''
        df_metrics_list = []
        for label in self.labeles:
            tmp_df = self.FullDataset.loc[self.FullDataset['label'] == label]
            tmp_df_metrics = pd.DataFrame(data = np.zeros((1, 4)), columns = ["target", "R2", "Theil", "Correlation"])
            target_true = tmp_df["target"].to_numpy()
            if target_type == "":
                target_predicted = np.dot(tmp_df[self.parameters[1 : ]].to_numpy(), self.solution_main).reshape(1, -1)[0]
            else:
                target_predicted = tmp_df["target"+target_type].to_numpy()
            print(f"r2_score для {label}:", r2_score(target_true, target_predicted))
            #print(tmp_df['target'].to_numpy())
            print(f"Индекс Тэйла для {label}:", self.__Theil(target_true, target_predicted))
            print(f"Корреляция таргета и прогноза для {label}:", np.corrcoef(target_true, target_predicted)[1, 0])
            tmp_df_metrics.iloc[0, 0] = label
            tmp_df_metrics.iloc[0, 1] = r2_score(target_true, target_predicted)
            tmp_df_metrics.iloc[0, 2] = self.__Theil(target_true, target_predicted)
            tmp_df_metrics.iloc[0, 3] = np.corrcoef(target_true, target_predicted)[1, 0]
            df_metrics_list.append(tmp_df_metrics)

        print("\nМоделирование ряда ВВП")
        target_true = self.aggregated_data_df['ln_Y_X_const_Y_0']
        target_predicted = self.alpha*(np.log(self.aggregated_data_df['Q_L'] / list(self.aggregated_data_df['Q_L'])[0])) +\
                           self.beta*(np.log(self.aggregated_data_df['Q_K'] / list(self.aggregated_data_df['Q_K'])[0]))
        print("r2_score для ВВП:", r2_score(target_true, target_predicted))
        print("Индекс Тэйла для ВВП:", self.__Theil(target_true, target_predicted))
        print(self.alpha)
        tmp_df_metrics = pd.DataFrame(data = np.zeros((1, 4)), columns = ["target", "R2", "Theil", "Correlation"])
        tmp_df_metrics.iloc[0, 0] = "GDP_Dataset"
        tmp_df_metrics.iloc[0, 1] = r2_score(target_true, target_predicted)
        tmp_df_metrics.iloc[0, 2] = self.__Theil(target_true, target_predicted)
        tmp_df_metrics.iloc[0, 3] = np.corrcoef(target_true, target_predicted)[1, 0]
        df_metrics_list.append(tmp_df_metrics)
        df_metrics = pd.concat(df_metrics_list)
        df_metrics.to_excel("Metrics"+target_type+".xlsx")

    def MakePrediction(self):
        '''
        Спрогнозируем основные показатели
        '''
        self.FullDataset['target_predicted'] = np.dot(self.FullDataset[self.parameters[1 : ]], self.solution_main)
        self.FullDataset['target_predicted_new'] = 0
        
        # Q_X
        tmp_df = self.FullDataset.loc[self.FullDataset['label'] == 'Q_X_Dataset']
        correction_coeff = (tmp_df['target_predicted']/tmp_df['target']).mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'Q_X_Dataset', 'target_predicted_new'] =\
                            list(tmp_df['target_predicted']/correction_coeff)
        
        # Q_K
        tmp_df = self.FullDataset.loc[self.FullDataset['label'] == 'Q_K_Dataset']
        correction_coeff = (tmp_df['target_predicted'][ : -1]/tmp_df['target'][ : -1]).mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'Q_K_Dataset', 'target_predicted_new'] =\
                            list(tmp_df['target_predicted']/correction_coeff)

        # Q_L
        tmp_df = self.FullDataset.loc[self.FullDataset['label'] == 'Q_L_Dataset']
        correction_coeff = ((1 / tmp_df['target_predicted'])/tmp_df['target']).mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'Q_L_Dataset', 'target_predicted_new'] =\
                            list((1/tmp_df['target_predicted'])/correction_coeff)

        # W_X
        correction_coeff = (self.FullDataset.loc[self.FullDataset['label'] == 'W_X_Dataset', 'target_predicted']/\
                           self.FullDataset.loc[self.FullDataset['label'] == 'W_X_Dataset', 'target']).mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'W_X_Dataset', 'target_predicted_new'] =\
                            self.FullDataset.loc[self.FullDataset['label'] == 'W_X_Dataset', 'target_predicted']/correction_coeff

        # Z_X
        correction_coeff = (self.FullDataset.loc[self.FullDataset['label'] == 'Z_X_Dataset', 'target_predicted']/\
                           self.FullDataset.loc[self.FullDataset['label'] == 'Z_X_Dataset', 'target']).mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'Z_X_Dataset', 'target_predicted_new'] =\
                            self.FullDataset.loc[self.FullDataset['label'] == 'Z_X_Dataset', 'target_predicted']/correction_coeff

        # W_G
        self.FullDataset['tmp'] = 0
        self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'tmp'] = [100000*math.sqrt(x*(-1)) for x in\
                                                list(self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'target_predicted'])]
        self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'tmp'] = self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'tmp'] -\
            self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'target']
        shift_coeff = self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'tmp'].mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'target_predicted_new'] = [100000*math.sqrt(x*(-1)) - shift_coeff for x in\
                                                list(self.FullDataset.loc[self.FullDataset['label'] == 'W_G_Dataset', 'target_predicted'])]
        
        # W_L
        self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'tmp'] = np.log(self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'target_predicted'])
        self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'tmp'] = self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'tmp']/\
                                                                               self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'target']
        correction_coeff = self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'tmp'].mean()
        self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'target_predicted_new'] =\
                        np.log(self.FullDataset.loc[self.FullDataset['label'] == 'W_L_Dataset', 'target_predicted'])/correction_coeff

        # R
        self.FullDataset.loc[self.FullDataset['label'] == 'R_Dataset', 'target_predicted_new'] = self.FullDataset.loc[self.FullDataset['label'] == 'R_Dataset', 'target_predicted']

        self.FullDataset.to_excel("FullData.xlsx")




        
        

em = EconModel('Data/UK_dataset.xls')
em.CreateDatasets()
em.Optimize()
em.BackTest()
em.MakePrediction()
em.BackTest("_predicted_new")