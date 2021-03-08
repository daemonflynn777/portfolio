import numpy as np
import scipy as sp

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
        self.T_max = 100
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
        print(prev_point)
        positions = np.arange(prev_point.shape[0])
        np.random.shuffle(positions)
        pos1 = positions[0]
        pos2 = positions[1]
        prev_point[[pos1, pos2]] = prev_point[[pos2, pos1]]
        print(prev_point)

    def OptimizeRoute(self):
        initial_point = self.start_point


    def test(self):
        self.__generate_next_point(self.start_point)
        


# M A I N
cities_file = open("cities.txt", "r")
tsp = TSP(10, cities_file)
tsp.test()