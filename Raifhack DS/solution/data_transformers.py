import pandas as pd
from pandas_profiling import ProfileReport
import os.path



data = pd.read_csv("../data/train.csv", low_memory = False)

#if os.path.exists("DataReport.html") == False:
#    print("Creating Data Report")
#    profile = ProfileReport(data, title = "Data Stats")
#    profile.to_file("DataReport.html")

