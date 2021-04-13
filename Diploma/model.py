import pandas as pd
import numpy as np

class EconModel():
    def __init__(self, dataset):
        self.aggregated_data_df = pd.read_excel(dataset)
        self.parameters = ["alpha", "beta", "a_X_XL", "a_X_XO", "b_X_XK",
                           "mu_K_X"]
        self.aggregated_data_df["ln_Y_X_const_Y_0"] = np.log(self.aggregated_data_df["Y_X_const"] / list(self.aggregated_data_df["Y_X_const"])[0])
        self.aggregated_data_df['dQ_X'] = np.diff(self.aggregated_data_df['Q_X'], append = self.aggregated_data_df['Q_X'][-1 : ])
        self.aggregated_data_df['Q_L'] = self.aggregated_data_df['employment_rate'] * self.aggregated_data_df['population']
        self.aggregated_data_df['dQ_L'] = np.diff(self.aggregated_data_df['Q_L'], append = self.aggregated_data_df['Q_L'][-1 : ])
        self.aggregated_data_df['dW_X'] = np.diff(self.aggregated_data_df['W_X'], append = self.aggregated_data_df['W_X'][-1 : ])
        self.aggregated_data_df['dQ_K'] = np.diff(self.aggregated_data_df['Q_K'], append = self.aggregated_data_df['Q_K'][-1 : ])
        self.aggregated_data_df['Z_X'] = self.aggregated_data_df['Z_X_part'] * self.aggregated_data_df['Y_X_const']
        self.aggregated_data_df['dZ_X'] = np.diff(self.aggregated_data_df['Z_X'], append = self.aggregated_data_df['Z_X'][-1 : ])
        self.aggregated_data_df['dW_G'] = np.diff(self.aggregated_data_df['W_G'], append = self.aggregated_data_df['W_G'][-1 : ])
        self.aggregated_data_df['dR'] = np.diff(self.aggregated_data_df['R'], append = self.aggregated_data_df['R'][-1 : ])
        self.aggregated_data_df['w_hat'] = 1 / self.aggregated_data_df['w']

    def CreateDatasets(self):
        '''
        Создадим датасет для идентификации
        параметров производственной функции (ВВП)
        '''
        #ln_Y_0 = list(self.aggregated_data_df["ln_Y_X_const"])[0]
        Q_L0_X = list(self.aggregated_data_df['Q_L'])[0]
        Q_K0_X = list(self.aggregated_data_df['Q_K'])[0]
        self.GDP_Dataset = pd.DataFrame(data = np.zeros((20, 3)), columns = ['alpha', 'beta', 'target'])
        self.GDP_Dataset['target'] = self.aggregated_data_df['ln_Y_X_const_Y_0'] -\
                                     np.log(self.aggregated_data_df['Q_K'] / list(self.aggregated_data_df['Q_K'])[0])
        self.GDP_Dataset['alpha'] = np.log(self.aggregated_data_df['Q_L'] / list(self.aggregated_data_df['Q_L'])[0]) -\
                                    np.log(self.aggregated_data_df['Q_K'] / list(self.aggregated_data_df['Q_K'])[0])

        '''
        Объединение нормативов (параметров)
        для всех остальных временных рядов
        '''
        self.parameters = ["target", "a_X_XL", "a_X_XO", "b_X_XK", "mu_K_X", "b_L_XL", "a_L_X", "b_X_GX", "b_M_LO"]

        '''
        Создадим датасет для идентификации
        параметров изменения блага производственного сектора
        '''
        self.Q_X_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_X_Dataset['target'] = self.aggregated_data_df['Y_X_const'] - self.aggregated_data_df['dQ_X']
        self.Q_X_Dataset['a_X_XL'] = self.aggregated_data_df['Q_X']
        self.Q_X_Dataset['a_X_XO'] = self.aggregated_data_df['Q_X']
        self.Q_X_Dataset['b_X_XK'] = self.aggregated_data_df['W_X'] / self.aggregated_data_df['p_X_X']
        for label in self.parameters - ['target', 'a_X_XL', 'a_X_XO', 'b_X_XK']:
            self.Q_X_Dataset[label] = 0

        '''
        Создадим датасет для идентификации
        параметров изменения запаса капитала (Q_K_X)
        '''
        self.Q_K_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_K_Dataset['target'] = self.aggregated_data_df['dQ_K']
        self.Q_K_Dataset['b_X_XK'] = self.aggregated_data_df['W_X'] / self.aggregated_data_df['p_X_X']
        self.Q_K_Dataset['mu_K_X'] = self.aggregated_data_df['Q_K'] * (-1)
        for label in self.parameters - ['target', 'b_X_XK', 'mu_K_X']:
            self.Q_K_Dataset[label] = 0
        
        '''
        Создадим датасет для идентификации
        параметров изменения запаса труда (Q_L_X)
        '''
        self.Q_L_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Q_L_Dataset['target'] = self.aggregated_data_df['dQ_L']
        self.Q_L_Dataset['b_L_XL'] = self.aggregated_data_df['W_X'] / self.aggregated_data_df['s_L_X']
        self.Q_L_Dataset['a_L_X'] = self.aggregated_data_df['Q_L'] * (-1)
        for label in self.parameters - ['target', 'b_L_XL', 'a_L_X']:
            self.Q_L_Dataset[label] = 0

        '''
        Создадим датасет для идентификации
        параметров изменения запаса денег
        в производственном секторе (W_X)
        '''
        self.W_X_dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        n_1 = self.aggregated_data_df['n_1']
        n_2 = self.aggregated_data_df['n_2']
        n_4 = self.aggregated_data_df['n_4']
        n_5 = self.aggregated_data_df['n_5']
        self.W_X_Dataset['target'] = self.aggregated_data_df['dW_X'] - (self.aggregated_data_df['T_GX'] + self.aggregated_data_df['C_BX'])
        self.W_X_Dataset['a_X_XL'] = (self.aggregated_data_df['p_X_L']*self.aggregated_data_df['Q_X_X'])*\
                                        (1 - self.aggregated_data_df['n_2'] - self.aggregated_data_df['n_1'] +\
                                        self.aggregated_data_df['n_1']*self.aggregated_data_df['n_2'])
        self.W_X_Dataset['a_X_XO'] = self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']*(1 - n_1 - n_2 - n_5 - n_1*n_2*n_5 +\
                                                                                                      n_1*n_5 + n_2*n_5 + n_1*n_2)
        self.W_X_Dataset['b_L_XL'] = self.aggregated_data_df['W_X'] * (n_1 + n_2*n_4 + n_1*n_4 - n_4 - n_1*n_2*n_4 - 1)
        self.W_X_Dataset['b_X_GX'] = self.aggregated_data_df['W_G']*(n_1 + n_2 - n_1*n_2)
        for label in self.parameters - ['target', 'a_X_XL', 'a_X_XO', 'b_L_XL', 'b_X_GX']:
            self.W_X_Dataset[label] = 0

        '''
        Создадим датасет для идентификации
        параметров изменения задолженности
        производственного сектора (Z_X)
        '''
        self.Z_X_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.Z_X_Dataset['target'] = self.aggregated_data_df['dZ_X'] - self.aggregated_data_df['H_XB']
        self.Z_X_Dataset['a_X_XO'] = (1 - self.aggregated_data_df['ksi'])*(1/self.aggregated_data_df['ksi'])*self.aggregated_data_df['w']*\
                                     self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']
        self.Z_X_Dataset['b_M_LO'] = (self.aggregated_data_df['ksi'] - 1)*(1/self.aggregated_data_df['ksi'])*self.aggregated_data_df['W_L']
        self.Z_X_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X'] * (-1)
        for label in self.parameters - ['target', 'a_X_XO', 'b_M_LO', 'b_Z_XB']:
            self.Z_X_Dataset[label] = 0

        '''
        Создадим датасет для идентификации
        параметров изменения индекса цен (p_X)
        '''

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
        self.W_G_Dataset['b_L_XL'] = self.aggregated_data_df['W_X']*(n_4 - n_1*n_4 + n1*n_2*n_4 - n_1 - n_2*n_4)
        self.W_G_Dataset['a_X_XL'] = self.aggregated_data_df['p_X_L']*self.aggregated_data_df['Q_X']*\
                                     (n_1 + n_2 + n_3 + n_1*n_2*n_3 - n_1*n_2 - n_2*n_3 - n_1*n_3)
        self.W_G_Dataset['b_Z_XB'] = self.aggregated_data_df['W_X']*(n_1*n_2 - n_1 - n_2)
        for label in self.parameters - ['target', 'b_M_LO', 'a_X_XO', 'b_L_XL', 'a_X_XL', 'b_Z_XB']:
            self.W_G_Dataset[label] = 0

        '''
        Создадим датасет для идентификации
        параметров изменения золотовалюиных
        резервов
        '''
        self.R_Dataset = pd.DataFrame(data = np.zeros((20, len(self.parameters))), columns = self.parameters)
        self.R_Dataset['target'] = self.aggregated_data_df['dR']
        self.R_Dataset['a_X_XO'] = self.aggregated_data_df['p_X_O']*self.aggregated_data_df['Q_X']
        self.R_Dataset['b_M_LO'] = self.aggregated_data_df['W_L'] / self.aggregated_data_df['w_hat']
        for label in self.parameters - ['target', 'a_X_XO', 'b_M_LO']:
            self.R_Dataset[label] = 0

    def Optimize(self):
        pass

em = EconModel('Data/UK_dataset.xls')
em.CreateDatasets()