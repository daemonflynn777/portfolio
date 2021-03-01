import numpy as np
import scipy as sp

class TSP():
    def __init__(self, cities_amount = 50):
        self.cities = np.random.uniform(0.0, 1.0, (2, cities_amount))*10 #cities[0] - координата X, cities[1] - координата Y
        print(self.cities)

tsp = TSP(10)