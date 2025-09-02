# 1: Import Libraries
#import pandas as pd
#import numpy as np

# 2: Download Data
#data = pd.read_excel("Data/EDA Data V0.xlsx")
#print(data.head(10))

# 1: Import Libraries
import pandas as pd

# 2: Download Data
data = pd.read_csv("Data_EXPLORATION.csv")
#Prints (rows,columns)
print("Shape:", data.shape)

missing_count = data.isna().sum() #Finds the sum of N/A values for each column
missing_percent = (data.isnull().mean()*100).round(2) #Finds the percentage of N/A values for each column

#Table for % of missing values
summary = pd.DataFrame({
        "Missing Count": missing_count, #Total # of missing values per column
        "Missing %": missing_percent    #Percentage of missing values per column
        }).sort_values("Missing %", ascending=False) #Sorts the values in ascending order

print("\nTOTAL % OF MISSING VALUES PER ATTRIBUTE")
print(summary.to_string())
