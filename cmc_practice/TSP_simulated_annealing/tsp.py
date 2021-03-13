import numpy as np
import math
import matplotlib.pyplot as plt
import io
import numbers

class TSP():
    def __init__(self, cities_input = 10):
        if isinstance(cities_input, int):
            if cities_input >= 2:
                self.cities = np.random.uniform(0.0, 1.0, (2, cities_input))*10 #cities[0] - координата X, cities[1] - координата Y
            else:
                raise ValueError("Неправильное количество городов. Их должно быть больше двух.")
        
        elif isinstance(cities_input, list):
            try:
                self.cities = np.array(cities_input)
            except:
                raise ValueError("Проверьте данные. Списиок координат должен быть двумерным и содержать одинковое число (>= 2) координат X и Y.")
        
        elif isinstance(cities_input, np.ndarray):
            if cities_input.shape[0] == 2 and cities_input.shape[1] >= 2:
                self.cities = cities_input.copy()
            else:
                raise ValueError("Проверьте данные. Массив координат должен быть двумерным и содержать одинковое число (>= 2) координат X и Y.")
        
        elif isinstance(cities_input, io.IOBase):
            cities_x = list(map(float, cities_input.readline().split()))
            cities_y = list(map(float, cities_input.readline().split()))
            if len(cities_x) != len(cities_y):
                raise ValueError(f"Количество координат по X и по Y должно совпадать.\nКоординат X: {len(cities_x)}, координат Y: {len(cities_y)}")
            self.cities = np.array([cities_x, cities_y], dtype = np.float_)
        
        else:
            raise ValueError("Неверный формат входных данных. Возможные форматы: int, list, NumPy ndarray, file.")

        self.start_point = np.array(list(range(1, self.cities.shape[1] + 1)))
        np.random.shuffle(self.start_point)
        self.T_max = 1000
        self.T_min = 0.01
        self.max_iter_no_change = self.cities.shape[1]*(self.cities.shape[1] - 1)/2

    def __objective_function(self, cities_order): #целевая функция суммарного расстояния между текущей последовательностью городов (закрытый метод класса)
        distance = 0.0
        cities_order = np.append(cities_order, cities_order[0 : 1])
        for i in range(cities_order.shape[0] - 1):
            prev_city = cities_order[i] - 1
            next_city = cities_order[i + 1] - 1
            distance += ((self.cities[0, next_city] - self.cities[0, prev_city])**2 + (self.cities[1, next_city] - self.cities[1, prev_city])**2)**0.5
        return distance

    def __temperature_decrease(self, iteration): #функция убывания температуры (закрытый метод класса)
        return self.T_max/iteration

    def __generate_next_point(self, prev_point): #функция генерации следующей последовательности городов (закрытый метод класса)
        positions = np.arange(prev_point.shape[0])
        np.random.shuffle(positions)
        tmp_point = prev_point.copy()
        pos1 = positions[0]
        pos2 = positions[1]
        tmp_point[[pos1, pos2]] = tmp_point[[pos2, pos1]]
        return tmp_point

    def __get_probability(self, curr_point, next_point, iteration): #вычисляем вероятность принятия следующей последовательности гоордов (закрытый метод класса)
        if self.__objective_function(next_point) < self.__objective_function(curr_point):
            return 1.0
        else:
            return math.exp(-(self.__objective_function(next_point) - self.__objective_function(curr_point))/self.__temperature_decrease(iteration))

    def OptimizeRoute(self): #поиск оптимальной последовательности городов
        iteration = 1
        initial_point = self.start_point
        next_point = self.start_point.copy()
        iter_no_change = 0
        while self.__temperature_decrease(iteration) >= self.T_min:
            curr_point = next_point.copy()
            next_point = self.__generate_next_point(curr_point)
            probability = self.__get_probability(curr_point, next_point, iteration)
            if probability < 1.0:
                ksi = abs(np.random.uniform(-1.0, 1.0, 1)[0])
                if ksi >= probability:
                    next_point = curr_point.copy()
                    iter_no_change += 1
            if (curr_point != next_point).all():
                iter_no_change = 0
            iteration += 1
        self.end_point = next_point.copy()
        print("Конечная последовательность городов:", self.end_point)
        print("Исходное суммарное расстояние:", self.__objective_function(initial_point))
        print("Конечное суммарное расстояние:", self.__objective_function(self.end_point))
        initial_matrix = np.array([[self.cities[0, i - 1], self.cities[1, i - 1]] for i in initial_point]).T
        end_matrix = np.array([[self.cities[0, i - 1], self.cities[1, i - 1]] for i in self.end_point]).T
        fig = plt.figure(figsize = (12, 4), num = "TSP")
        plt.subplot(1, 3, 1, title = "Города")
        plt.scatter(self.cities[0], self.cities[1])
        plt.subplot(1, 3, 2, title = "Начальный путь")
        plt.scatter(self.cities[0], self.cities[1])
        plt.plot(initial_matrix[0], initial_matrix[1])
        plt.subplot(1, 3, 3, title = "Оптимальный путь")
        plt.scatter(self.cities[0], self.cities[1])
        plt.plot(end_matrix[0], end_matrix[1])
        plt.show()