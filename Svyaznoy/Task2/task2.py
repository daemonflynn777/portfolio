import numpy as np
import pandas as pd


def GetData(FileName):
    input_file = open(FileName, 'r')
    matrix = []
    for line in input_file:
        matrix.append([float(num) for num in line.split()])
    matrix_np = np.array(matrix)
    columns = [f"x_{i}" for i in range(1, 11)]
    columns.append("y")
    dframe = pd.DataFrame(data = matrix_np, columns = columns)
    dframe.to_csv("task2.csv")
    return dframe


# M A I N
df = GetData("task2.txt")
#df = pd.read_csv("task2.csv")
