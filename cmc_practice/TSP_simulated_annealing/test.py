import tsp
import numpy as np
import time
import matplotlib.pyplot as plt

def TestTSP():
    tsp_test = tsp.TSP()
    curr_time = time.time()
    tsp_test.OptimizeRoute()
    print(f"Test 1 passed.\nElapsed time: {time.time() - curr_time} seconds\n")

    cities_file = open("cities.txt", "r")
    tsp_test = tsp.TSP(cities_file)
    curr_time = time.time()
    tsp_test.OptimizeRoute()
    print(f"Test 2 passed.\nElapsed time: {time.time() - curr_time} seconds\n")
    cities_file.close()

    cities_list = [[1.5, 7.2, 5.2, 9.9], [2.28 , 6.66, 8.34, 4.04]]
    tsp_test = tsp.TSP(cities_list)
    curr_time = time.time()
    tsp_test.OptimizeRoute()
    print(f"Test 3 passed.\nElapsed time: {time.time() - curr_time} seconds\n")

    cities_array = np.array([[2.28 , 6.66, 8.34, 4.04], [1.5, 7.2, 5.2, 9.9]])
    tsp_test = tsp.TSP(cities_array)
    curr_time = time.time()
    tsp_test.OptimizeRoute()
    print(f"Test 4 passed.\nElapsed time: {time.time() - curr_time} seconds\n")

def ElaplesTime():
    cities_count = [5*i for i in range(1, 11)]
    time_for_cities_set = [1.4, 7.0, 21.0, 26.0, 31.0, 38.0, 41.0, 47.0, 53.0, 57.0]
    fig = plt.figure(figsize = (6, 4), num = "Elapsed time")
    plt.scatter(x = cities_count, y = time_for_cities_set, c = "purple")
    plt.xlabel("Количество городов")
    plt.ylabel("Затраченное время, секунд")
    plt.show()

# M A I N
TestTSP()
#ElaplesTime()