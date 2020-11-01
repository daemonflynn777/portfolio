import numpy as np
from numpy import linalg as LA
from math import sqrt
from tkinter import *
from tkinter.ttk import Checkbutton
from tkinter import scrolledtext


class Functional():
    def __init__(self, input_func, sphere_center, sphere_rad, start_point, tol, max_iter, input_func_str = "функция многих переменных", silent = 1):
        self.func = input_func
        #self.func1d = input_func1d
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
        print("Ограничения на множество:")
        #for i in range(len(start_point)):
        #    print("%f <= x_%i <= %f" %(self.limits[i][0], i, self.limits[i][1]))

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
        #eps = (0.01)**self.precision
        eps = self.precision**2
        delta = (c - a)/precision
        x = a
        values = []
        while x <= c:
            values.append(self.func(self.u_k + x*(self.u_k_line - self.u_k)))
            x += delta
        return a + delta*values.index(min(values)), min(values)

        #delta = 0.1
        #a_cap = 1
        #while self.func(self.u_k + a_cap*(self.u_k_line - self.u_k)) >= self.func(self.u_k):
        #    a_cap *= delta
        #return a_cap, self.func(self.u_k + a_cap*(self.u_k_line - self.u_k))



    def Optimize(self, textbox, axes, plotbox): # вычисляет оптимальную точку
        #eps = (0.1)**(self.precision)
        eps = self.precision
        #coefs = self.Gradient(self.u_k)
        #self.u_k_line = np.array([self.limits[i][int(coefs[i] <= 0)] for i in range(len(self.limits))])
        self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/LA.norm(self.Gradient(self.u_k))
        step = 1
        #self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/sqrt(np.dot(self.Gradient(self.u_k), self.Gradient(self.u_k)))
        #print(self.u_k_line)
        #print(self.sph_center.tolist())
        self.alpha_k = self.FindMin()[0]
        #elf.alpha_k = 2.0/(step + 2.0)
        u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()
        func_vals = [self.func(self.u_k)]
        if self.silent == 0:
            #print("\nТекущее значение u_k:", self.u_k)
            #print("Текущее значение u_k_с_чертой:", self.u_k_line)
            #print("Текущее значение alpha_k:", self.alpha_k)
            #print("Текущее значение u_k+1:", u_k_next)
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
        #while LA.norm(np.array(u_k_next) - self.u_k) >= eps:
            step += 1
            self.u_k = u_k_next
            func_vals.append(self.func(self.u_k))
            #coefs = self.Gradient(self.u_k)
            self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/LA.norm(self.Gradient(self.u_k))
            #self.u_k_line = self.sph_center - sqrt(self.sph_rad)*self.Gradient(self.u_k)/sqrt(np.dot(self.Gradient(self.u_k), self.Gradient(self.u_k)))
            #self.alpha_k = 2.0/(step + 2.0)
            self.alpha_k = self.FindMin()[0]
            u_k_next = (self.u_k + self.alpha_k*(self.u_k_line - self.u_k)).tolist()

            axes.cla()
            axes.set_ylim([0.0 , max(func_vals)]) 
            axes.grid()
            #axes.set_xlim(min(func_vals), max(func_vals))
            
            axes.plot(range(1, step + 1), func_vals, color = 'purple')
            plotbox.draw()

            if self.silent == 0:
                #print("\nТекущее значение u_k:", self.u_k)
                #print("Текущее значение u_k_с_чертой:", self.u_k_line)
                #print("Текущее значение alpha_k:", self.alpha_k)
                #print("Текущее значение u_k+1:", u_k_next)
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
        #print("\nОптимизация завершена")
        #print("Минимальное знаение функционала:", self.func(u_k_next))
        #print("Точка минимума:", u_k_next)
        func_vals.append(self.func(u_k_next))
        print(func_vals)

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