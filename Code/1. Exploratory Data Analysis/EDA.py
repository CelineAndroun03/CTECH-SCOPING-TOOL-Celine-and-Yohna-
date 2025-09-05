
# ##################################################################
#     #1) Load the Excel dataset
# ##################################################################

# 1: Import Libraries
from scipy.stats import skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.api.types as ptypes
import warnings

warnings.filterwarnings("ignore") # Ignore python warnings
pd.set_option('display.max_columns', None) # Settings to print all columns in the dataframe
pd.set_option('display.max_rows', None) # Settings to print all rows in the dataframe

# 2: Download Data
data = pd.read_csv("Data_EXPLORATION.csv")
print("Shape:", data.shape) #Prints (rows,columns)

# ##################################################################
#     #2) Gather Intital Information
# ##################################################################
print("##########################################")
print("##### Dataset Preliminary Informaion #####")
print("##########################################" + "\n")

print("Dataframe head(5):")
print(data.head(5)) # Prints the first 5 rows
print()

print("Dataframe shape:")
print("Number of columns:", str(data.shape[1])) # Prints (rows,columns)
print("Number of rows:", str(data.shape[0]) + '\n') # Prints (rows,columns)

print("Descriptive Statistics:")
print(data.describe()) # Generate descriptive statistics
print()

print("Dataframe dtypes:")
print(data.dtypes) # Return the dtypes in the DataFrame
print()

print("Duplicated Rows: " + str(data.duplicated().sum()) + ("\n")) # Return boolean Series denoting duplicate rows

# ##################################################################
#     #3) Early Deletion(Columns That are Not Valuable)
# ##################################################################

data = data.drop(columns=[ #After visualizing data and seeing what is valuable we can delete more in later steps,
                           #As of right now, I don't think these attributes are valuable at all; however, let me know if this deletion step is too early
    "Final Field Name",
    "Filename (2023/2024)",
    "CPQ Product SKUs",
    "CCN_Data Hub",
    "Created By",
    "Created by: First Name",
    "Created by: Last Name",
    "Created by Location",
    "Internal Sales Rep",
    "Product_Name",
    "Opportunity Number",
    "Quote Number",
    "Project Number",
    "Company"
])
# #print(df_clean.columns.tolist())

# ##################################################################
#     #4) Missing Percentage / Distribution type 
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

# ##################################################################
#     #5) Data Visualization
# ##################################################################

#Plot the frequency of each investigation type
plt.figure(figsize=(8, 6))
ax = data["type_of_investigation"].value_counts().plot(kind="bar", color='skyblue')
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel('Type of Investigation')
plt.ylabel('Count')
plt.title('Frequency of Each Investigation Type')
plt.legend(['Investigation Type'])
plt.tight_layout()
plt.show()

# #Plot the average of actual hours for each of these types (total, lab, eng)
# #(How is it calculated? For example, lets take one investigation type, Full investigation,
# #Look at all the projects under that investigation type and add all of the "total actual hrs" up.
# #Count how many there are and divide them to see the average hrs spent in that investigation type)

#Calcuates the average total actual hours for each investigation type
plt.figure(figsize=(8, 6))
avg_hours = data.groupby("type_of_investigation") ["total_actual_hrs"].mean()
ax = avg_hours.plot(kind="bar", color='skyblue')
ax.bar_label(ax.containers[0])
plt.xlabel('Type of Investigation')
plt.ylabel('Count')
plt.title('Average Total Actual Hours by Investigation Type')
plt.legend(['Investigation Type'])
plt.show()

#Calcuates the average total lab actual hours for each investigation type
plt.figure(figsize=(8, 6))
avg_labHours = data.groupby("type_of_investigation") ["lab_actual_hrs"].mean()
ax1 = avg_labHours.plot(kind="bar", color='skyblue')
ax1.bar_label(ax1.containers[0]) #when you draw bars matplotlib puts them into containers, takes the bar and adds the Num on top
plt.xlabel('Type of Investigation')
plt.ylabel('Count')
plt.title('Average Lab Actual Hours by Investigation Type')
plt.legend(['Investigation Type'])
plt.show()

#Calcuates the average total actual eng hours for each investigation type
plt.figure(figsize=(8, 6))
avg_engHours = data.groupby("type_of_investigation") ["eng_actual_hrs"].mean()
ax2 = avg_engHours.plot(kind="bar", color='skyblue')
ax2.bar_label(ax2.containers[0])
plt.xlabel('Type of Investigation')
plt.ylabel('Count')
plt.title('Average Engineering Actual Hours by Investigation Type')

plt.show()
print()
print()

# ##################################################################
# #6) correlation between the variables and the targets
# ##################################################################

print("############################### Lab Actual Hours Correlation to Other Attributes #################################")
corr = data.corr(numeric_only=True)
print(corr["lab_actual_hrs"].sort_values(ascending=False))
print("############################### Eng Actual Hours Correlation to Other Attributes #################################")
print(corr["eng_actual_hrs"].sort_values(ascending=False))

#I need to look into Engineering and lab actual hours more in depth, since they may be influenced by different variables
#or the same variables but in different ways


# corr = data.corr(numeric_only=True)
# drop_cols = [col for col in corr.columns if "hrs" in col.lower()]
# lab_corr = corr ["lab_actual_hrs"].drop(drop_cols).sort_values(ascending=False)
# eng_corr = corr ["eng_actual_hrs"].drop(drop_cols).sort_values(ascending=False)
# comparison = pd.DataFrame({
#     "lab_Actual_Hrs": lab_corr,
#     "eng_Actual_Hrs": eng_corr,
# })
# print(comparison)