import numpy as np
from numpy import linalg as LA
from math import sqrt

class Functional():
    def __init__(self, input_func, input_func1d, input_lims, dimensions, input_func_str = "функция многих переменных", silent = 1):
        self.func = input_func
        self.func1d = input_func1d
        self.input_func_str = input_func_str
        self.limits = input_lims
        self.u_k = np.array([np.random.uniform(input_lims[i][0], input_lims[i][1], 1) for i in range(dimensions)]).reshape(dimensions) # начальная точка, каждая координата которой удовлетворяет ограничениям на множество
        self.u_k_line = np.zeros(dimensions)
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
            grad.append((self.func(dimension_f) - self.func(dimension_b)) / (2 * h)) # центральная производная
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

    def BrentComb(self, a = 0, c = 1, eps = 0.001): # комбинированный метод Брента минимизации ф-ии одной переменной
        K = (3 - sqrt(5)) / 2
        x = (a + c)/2
        w = (a + c)/2
        v = (a + c)/2
        u = 2*c
        f_x = self.func1d(x)
        f_w = self.func1d(w)
        f_v = self.func1d(v)
        while abs(u - x) >= eps:
            if ((x != w and w != v and x != v) and (f_x != f_w and f_w != f_v and f_x != f_v)):
                u = x - (((x - v)**2)*(f_x - f_w) - ((x - w)**2)*(f_x - f_v))/(2*((x - v)*(f_x - f_w) - (x - w)*(f_x - f_v)))
            if ((u >= a + eps) and (u <= c - eps) and abs(u - x) < g/2):
                d = abs(u - x)
            else:
                if x < (c - a)/2:
                    u = x + K*(c - x) # золотое сечение на [x, c]
                    d = c - x
                else:
                    u = x - K*(x - a) # золотое сечение на [a, x]
                    d = x - a
            if abs(u - x) < eps:
                u = x + eps*np.sign(u - x)
            f_u = self.func1d(u)
            if f_u <= f_x:
                if u >= x:
                    a = x
                else:
                    c = x
                v = w
                w = x
                x = u
                f_v = f_w
                f_w = f_x
                f_x = f_u
            else:
                if u >= x:
                    c = u
                else:
                    a = u
                if ((f_u <= f_w) or (w == x)):
                    v = w
                    w = u
                    f_v = f_w
                    f_w = f_u
                elif ((f_u <= f_v) or (v == x) or (v == w)):
                    v = u
                    f_v = f_u
        return u, f_u

    def Optimize(self, eps = 0.001): # вычисляет оптимальную точку
        coefs = self.Gradient(self.u_k)
        self.u_k_line = np.array([self.limits[i][int(coefs[i] <= 0)] for i in range(len(self.limits))])
        self.alpha_k = self.BrentComb()[0]
        u_k_next = self.u_k + self.alpha_k*(self.u_k_line - self.u_k)
        if self.silent == 0:
            print("\nТекущее значение u_k:", self.u_k)
            print("Текущее значение u_k_с_чертой:", self.u_k_line)
            print("Текущее значение alpha_k:", self.alpha_k)
            print("Текущее значение u_k+1:", u_k_next)
        while LA.norm(u_k_next - self.u_k) >= eps:
            self.u_k = u_k_next
            coefs = self.Gradient(self.u_k)
            self.u_k_line = np.array([self.limits[i][int(coefs[i] <= 0)] for i in range(len(self.limits))])
            self.alpha_k = self.BrentComb()[0]
            u_k_next = self.u_k + self.alpha_k*(self.u_k_line - self.u_k)
            if self.silent == 0:
                print("\nТекущее значение u_k:", self.u_k)
                print("Текущее значение u_k_с_чертой:", self.u_k_line)
                print("Текущее значение alpha_k:", self.alpha_k)
                print("Текущее значение u_k+1:", u_k_next)
        print("\nОптимизация завершена")
        print("Минимальное знаение функционала:", self.func(u_k_next))
        print("Точка минимума:", u_k_next)