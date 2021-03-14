import tsp
import numpy as np

def TestTSP():
    tsp_test = tsp.TSP()
    tsp_test.OptimizeRoute()
    print("Test 1 passed.\n")

    cities_file = open("cities.txt", "r")
    tsp_test = tsp.TSP(cities_file)
    tsp_test.OptimizeRoute()
    print("Test 2 passed.\n")
    cities_file.close()

    cities_list = [[1.5, 7.2, 5.2, 9.9], [2.28 , 6.66, 8.34, 4.04]]
    tsp_test = tsp.TSP(cities_list)
    tsp_test.OptimizeRoute()
    print("Test 3 passed.\n")

    cities_array = np.array([[2.28 , 6.66, 8.34, 4.04], [1.5, 7.2, 5.2, 9.9]])
    tsp_test = tsp.TSP(cities_array)
    tsp_test.OptimizeRoute()
    print("Test 4 passed.\n")

# M A I N
TestTSP()