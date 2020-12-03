import numpy as np
import pandas as pd
from numpy import linalg as LA
from math import sqrt
from tkinter import *
from tkinter.ttk import Checkbutton
from tkinter import scrolledtext


class Functional():
    def __init__(self, input_func, sphere_center, sphere_rad, start_point, tol, max_iter, input_func_str = "функция многих переменных", silent = 1):
        self.func = input_func
        self.input_func_str = input_func_str
        self.sph_center = np.array(sphere_center)
        self.sph_rad = sqrt(sphere_rad)
        self.u_k = np.array(start_point) # начальная точка, каждая координата которой удовлетворяет ограничениям на множество
        self.u_k_line = np.zeros(len(start_point)).tolist()
        self.alpha_k = 0
        self.precision = tol
        self.max_iter = max_iter
        self.silent = silent
        print("Минимизируемый функционал:", self.input_func_str)

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

    def FindMin(self, a = 0.0, c = 1.0, precision = 10000): # метод покрытий для минимизации ф-ии одной переменной
        eps = self.precision**1
        delta = (c - a)/precision
        x = a
        values = []
        while x <= c:
            values.append(self.func(self.u_k + x*(self.u_k_line - self.u_k)))
            x += delta
        return a + delta*values.index(min(values)), min(values)

    def Optimize(self, textbox = None, axes = None, plotbox = None): # вычисляет оптимальную точку

        clmns = ["alpha_k"] + [f"x_k_{i}" for i in range(len(self.sph_center))] + [f"x_k_line_{i}" for i in range(len(self.sph_center))] + ["f_k"]
        u_k_first = self.u_k.tolist()

        step = 1
        eps = self.precision
        self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/LA.norm(self.Gradient(self.u_k))
        self.alpha_k = self.FindMin()[0]

        iter_data = [self.alpha_k] + self.u_k.tolist() + self.u_k_line.tolist() + [self.func(self.u_k)]
        df_data = [iter_data]

        u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()
        func_vals = [self.func(self.u_k)]
        if self.silent == 0:
            if textbox is not None:
                textbox.insert(INSERT, "Итерация ")
                textbox.insert(INSERT, step)
                textbox.insert(INSERT, "\n")
                textbox.insert(INSERT, self.u_k)
                textbox.insert(INSERT, "\n")
                textbox.insert(INSERT, self.u_k_line)
                textbox.insert(INSERT, "\n")
                textbox.insert(INSERT, self.alpha_k)
                textbox.insert(INSERT, "\n")
                textbox.insert(INSERT, u_k_next)
                textbox.insert(INSERT, "\n")
                textbox.insert(INSERT, "\n")
                textbox.update_idletasks()
        while (abs(self.func(np.array(u_k_next)) - self.func(self.u_k)) >= eps) and (step < self.max_iter):
            step += 1
            self.u_k = u_k_next
            func_vals.append(self.func(self.u_k))
            self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/LA.norm(self.Gradient(self.u_k))
            self.alpha_k = self.FindMin()[0]

            iter_data = [self.alpha_k] + self.u_k + self.u_k_line.tolist() + [self.func(self.u_k)]
            df_data.append(iter_data)

            u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()

            if (axes is not None) and (plotbox is not None):
                axes.cla()
                axes.set_ylim([0.0 , max(func_vals)]) 
                axes.grid()
                axes.plot(range(1, step + 1), func_vals, color = 'purple')
                plotbox.draw()

            if self.silent == 0:
                if textbox is not None:
                    textbox.insert(INSERT, "Итерация ")
                    textbox.insert(INSERT, step)
                    textbox.insert(INSERT, "\n")
                    textbox.insert(INSERT, self.u_k)
                    textbox.insert(INSERT, "\n")
                    textbox.insert(INSERT, self.u_k_line)
                    textbox.insert(INSERT, "\n")
                    textbox.insert(INSERT, self.alpha_k)
                    textbox.insert(INSERT, "\n")
                    textbox.insert(INSERT, u_k_next)
                    textbox.insert(INSERT, "\n")
                    textbox.insert(INSERT, "\n")
                    textbox.update_idletasks()
        print("Оптимизация завершена", "\nПосмотреть результаты можно в Excel-файле\n")
        func_vals.append(self.func(u_k_next))

        df = pd.DataFrame(df_data, columns = clmns)
        try:
            df.to_excel("Optimization.xlsx")
        except:
            pass

        if (axes is not None) and (plotbox is not None) and (textbox is not None):
            axes.cla()
            axes.set_ylim([0.0 , max(func_vals)]) 
            axes.grid()
            axes.plot(range(1, step + 2), func_vals, color = 'purple')
            plotbox.draw()

            textbox.insert(INSERT, "Оптимизация завершена\n")
            textbox.insert(INSERT, "Минимальное знаение функционала: ")
            textbox.insert(INSERT, self.func(u_k_next))
            textbox.insert(INSERT, "\n")
            textbox.insert(INSERT, "Точка минимума: ")
            textbox.insert(INSERT, u_k_next)
            textbox.update_idletasks()

        return u_k_first + u_k_next + [func_vals[len(func_vals) - 1]] + [step]