import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

class TSP():
    def __init__(self, cities_amount = 50, cities_input = None):
        if cities_input == None:
            self.cities = np.random.uniform(0.0, 1.0, (2, cities_amount))*10 #cities[0] - координата X, cities[1] - координата Y
            print(self.cities)
        else:
            cities_x = list(map(float, cities_input.readline().split()))
            cities_y = list(map(float, cities_input.readline().split()))
            if len(cities_x) != len(cities_y):
                raise ValueError(f"Количество координат по X и по Y должно совпадать.\nКоординат X: {len(cities_x)}, координат Y: {len(cities_y)}")
            self.cities = np.array([cities_x, cities_y], dtype = np.float_)
        self.start_point = np.array(list(range(1, self.cities.shape[1] + 1)))
        np.random.shuffle(self.start_point)
        self.T_max = 1000
        self.T_min = 0.01

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
        #pos3 = positions[2]
        #np.random.shuffle(tmp_point)
        #tmp_point[[min(pos1, pos2) : max(pos1, pos2)]] = np.flip(tmp_point[min(pos1, pos2) : max(pos1, pos2)])
        tmp_point1 = tmp_point[ : min(pos1, pos2)]
        tmp_point2 = np.flip(tmp_point[min(pos1, pos2) : max(pos1, pos2)])
        tmp_point3 = tmp_point[max(pos1, pos2) : ]
        tmp_point = np.append(tmp_point1, tmp_point2)
        tmp_point = np.append(tmp_point, tmp_point3)
        #tmp_point = np.roll(tmp_point, 13)
        #tmp_point = np.flip(tmp_point)
        return tmp_point

    def __get_probability(self, curr_point, next_point, iteration): #вычисляем вероятность принятия следующей последовательности гоордов (закрытый метод класса)
        if self.__objective_function(next_point) < self.__objective_function(curr_point):
            return 1.0
        else:
            return math.exp(-(self.__objective_function(next_point) - self.__objective_function(curr_point))/self.__temperature_decrease(iteration))

    def OptimizeRoute(self): #поиск оптимальной последовательности городов
        iteration = 1
        initial_point = self.start_point
        curr_point = self.start_point.copy()
        next_point = self.__generate_next_point(curr_point)
        probability = self.__get_probability(curr_point, next_point, iteration)
        while self.__temperature_decrease(iteration) >= self.T_min:
            print(probability)
            if probability < 1.0:
                ksi = abs(np.random.normal(0.0, 0.1, 1)[0])
                if ksi > probability:
                    print(iteration)
                    #next_point = curr_point.copy()
                    break
            iteration += 1
            curr_point = next_point.copy()
            next_point = self.__generate_next_point(curr_point)
            probability = self.__get_probability(curr_point, next_point, iteration)
        self.end_point = curr_point.copy()
        print("Конечная последовательность городов:", self.end_point)
        print("Исходное суммарное расстояние:", self.__objective_function(initial_point))
        print("Конечное суммарное расстояние:", self.__objective_function(self.end_point))
        initial_matrix = np.array([[self.cities[0, i - 1], self.cities[1, i - 1]] for i in initial_point]).T
        end_matrix = np.array([[self.cities[0, i - 1], self.cities[1, i - 1]] for i in self.end_point]).T
        fig = plt.figure(figsize = (12, 4), num = "TSP")
        plt.subplot(1, 3, 1)
        plt.scatter(self.cities[0], self.cities[1])
        plt.subplot(1, 3, 2)
        plt.scatter(self.cities[0], self.cities[1])
        plt.plot(initial_matrix[0], initial_matrix[1])
        plt.subplot(1, 3, 3)
        plt.scatter(self.cities[0], self.cities[1])
        plt.plot(end_matrix[0], end_matrix[1])
        #plt.show()

    def test(self):
        self.__generate_next_point(self.start_point)
        


# M A I N
cities_file = open("cities.txt", "r")
tsp = TSP(20)
tsp.OptimizeRoute()