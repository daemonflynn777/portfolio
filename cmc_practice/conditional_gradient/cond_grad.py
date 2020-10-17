import numpy as np
from numpy import linalg as LA
from math import sqrt

class Functional():
    def __init__(self, input_func, input_lims, dimensions, input_func_str = "функция многих переменных", silent = 1):
        self.func = input_func
        #self.func1d = input_func1d
        self.input_func_str = input_func_str
        self.limits = input_lims
        self.u_k = np.array([np.random.uniform(input_lims[i][0], input_lims[i][1], 1) for i in range(dimensions)]).reshape(dimensions) # начальная точка, каждая координата которой удовлетворяет ограничениям на множество
        self.u_k_line = np.zeros(dimensions).tolist()
        self.alpha_k = 0
        self.silent = silent
        print("Минимизируемый функционал:", self.input_func_str)
        print("Ограничения на множество:")
        for i in range(dimensions):
            print("%f <= x_%i <= %f" %(self.limits[i][0], i, self.limits[i][1]))

    def Gradient(self, point):
        grad = []
        dim = len(point)
        h = 0.0001 #обдумать этот моментик
        for pos, num in enumerate(point):
            dimension_f = np.array(point)
            dimension_b = np.array(point)
            dimension_f[pos] += h #шаг вперед
            dimension_b[pos] -= h #шаг назад
            grad.append((self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h)) # центральная производная
        return np.array(grad)

    def Hessian(self, point):
        hess = []
        dim = len(point)
        h = 0.0001 #обдумать этот моментик
        for pos1, num1 in enumerate(point):
            partial_derivative = []
            for pos2, num2 in enumerate(point):
                dimension_f = np.array(point)
                dimension_m = np.array(point)
                dimension_b = np.array(point)
                if pos1 == pos2:
                    dimension_f[pos1] = num1 + h #шаг вперед
                    dimension_m[pos1] = num1 #центральная точка
                    dimension_b[pos1] = num1 - h #шаг назад
                    partial_derivative.append((self.func(dimension_f.tolist()) + self.func(dimension_b.tolist()) - 2*self.func(dimension_m.tolist())) / (h**2))
                else:
                    dimension_f[pos1] += h #шаг вперед, координата 1
                    dimension_b[pos1] -= h #шаг назад, координата 1
                    dimension_f[pos2] += h #фиксируем координату 2
                    dimension_b[pos2] += h #фиксируем координату 2
                    der1 = (self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h) #производная по координате 1
                    dimension_f[pos1] = num1 + h
                    dimension_b[pos1] = num1 - h
                    dimension_f[pos2] -= h #шаг вперед, координата 2
                    dimension_b[pos2] -= h #шаг назад, координата 2
                    der2 = (der1 - (self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h)) / (1 * h) #производная по координате 2
                    partial_derivative.append(der2)
            hess.append(partial_derivative)
        return np.array(hess)

    def FindMin(self, a = 0.0, c = 1.0, flt_num = 3, precision = 1000000): # метод покрытий для минимизации ф-ии одной переменной
        eps = (0.01)**flt_num
        delta = (c - a)/precision
        x = a + eps
        values = []
        while x <= c - eps:
            values.append(self.func(self.u_k + x*(self.u_k_line - self.u_k)))
            x += delta
        return a + eps + delta*values.index(min(values)), min(values)



    def Optimize(self, flt_num = 3): # вычисляет оптимальную точку
        eps = (0.1)**flt_num
        coefs = self.Gradient(self.u_k)
        self.u_k_line = np.array([self.limits[i][int(coefs[i] <= 0)] for i in range(len(self.limits))])
        self.alpha_k = self.FindMin()[0]
        u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()
        if self.silent == 0:
            print("\nТекущее значение u_k:", self.u_k)
            print("Текущее значение u_k_с_чертой:", self.u_k_line)
            print("Текущее значение alpha_k:", self.alpha_k)
            print("Текущее значение u_k+1:", u_k_next)
        while round(LA.norm(np.array(u_k_next) - self.u_k), flt_num) >= eps:
            self.u_k = u_k_next
            coefs = self.Gradient(self.u_k)
            self.u_k_line = np.array([self.limits[i][int(coefs[i] <= 0)] for i in range(len(self.limits))])
            self.alpha_k = self.FindMin()[0]
            u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()
            if self.silent == 0:
                print("\nТекущее значение u_k:", self.u_k)
                print("Текущее значение u_k_с_чертой:", self.u_k_line)
                print("Текущее значение alpha_k:", self.alpha_k)
                print("Текущее значение u_k+1:", u_k_next)
        print("\nОптимизация завершена")
        print("Минимальное знаение функционала:", self.func(u_k_next))
        print("Точка минимума:", u_k_next)