# # 1: Import Libraries
# #import pandas as pd
# #import numpy as np

# # 2: Download Data
# #data = pd.read_excel("Data/EDA Data V0.xlsx")
# #print(data.head(10))

# ##################################################################
#     #1) Load the Excel dataset
# ##################################################################

# 1: Import Libraries
from scipy.stats import skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.api.types as ptypes

# 2: Download Data
data = pd.read_csv("Data_EXPLORATION.csv")
print("Shape:", data.shape) #Prints (rows,columns)

# ##################################################################
#     #2) Missing Percentage / Distribution type 
# ##################################################################
data["total_actual_hrs"] = pd.to_numeric(data["total_actual_hrs"], errors="coerce")
data["lab_actual_hrs"] = pd.to_numeric(data["lab_actual_hrs"], errors="coerce")
data["eng_actual_hrs"] = pd.to_numeric(data["eng_actual_hrs"], errors="coerce")
data["total_eng_hrs"] = pd.to_numeric(data["total_eng_hrs"], errors="coerce")

missing_count = data.isna().sum() #Finds the sum of N/A values for each column
missing_percent = (data.isna().mean()*100).round(2) #Finds the percentage of N/A values for each column

distribution = {} 
for col in data.columns: #Loops through every column name in the dataset
    if data[col].dropna().empty: #drop the missing values, and check if anything is left
        distribution[col] = "No data" #If nothing is left, No data
    elif not ptypes.is_numeric_dtype(data[col]):
        distribution[col] = "Categorical"
    else:
        s = skew(data[col].dropna()) #Selects the column and removes missing values
        if s > 1:
            distribution[col] = "Right skewed" #skew is positive
        if s < -1:
            distribution[col] = "Left skewed" #skew is negative
        else:
            distribution[col] = "Normal" #bell curve

#Table for % of missing values
summary = pd.DataFrame({
        "Data Type" : data.dtypes.astype(str), #Type of data
        "Missing Count": missing_count, #Total # of missing values per column
        "Missing %": missing_percent,    #Percentage of missing values per column
        "  Distribution": distribution
        }).sort_values("Missing %", ascending=False) #Sorts the values in ascending order

print("\nMISSING VALUES PER ATTRIBUTE + DISTRIBUTION")
print(summary.to_string())

##################################################################
    #3) Data Visualization
##################################################################


#Plot the frequency of each investigation type
plt.figure(figsize = (6,6))
data["type_of_investigation"].value_counts().plot(kind = "bar")
ax = data["type_of_investigation"].value_counts().plot(kind="bar", figsize=(6,6))
for container in ax.containers:
    ax.bar_label(container)
plt.show()

#lot the average of actual hours for each of these types (total, lab, eng)

#Average total actual hours per project by investigation type
avg_hours = data.groupby("type_of_investigation") ["total_actual_hrs"].mean()
ax = avg_hours.plot(kind="bar")
ax.bar_label(ax.containers[0])
plt.show()

#Average actual lab hours per project by investigation type
avg_labHours = data.groupby("type_of_investigation") ["lab_actual_hrs"].mean()
ax1 = avg_labHours.plot(kind="bar")
ax1.bar_label(ax1.containers[0])
plt.show()

#Average actual eng hours per project by investigation type
avg_engHours = data.groupby("type_of_investigation") ["eng_actual_hrs"].mean()
ax2 = avg_engHours.plot(kind="bar")
ax2.bar_label(ax2.containers[0])
plt.show()

##################################################################
#4) correlation between the variables and the targets
##################################################################