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
import seaborn as sns
from pandas.api.types import is_numeric_dtype

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

#print(data["CCN_Data Hub"].nunique())
#print(data["CCN_Data Hub"].value_counts())

# ##################################################################
#     #3) Early Deletion(Columns That are Not Valuable)
# ##################################################################

data = data.drop(columns=[ #I will leave this step for you since you have the best understanding of which columns are needed
                           
      



])
# #print(df_clean.columns.tolist())

# ##################################################################

# ##################################################################
#     #4) Missing Percentage / Distribution type 
# ##################################################################
data["Eng. SH"] = pd.to_numeric(data["Eng. SH"], errors="coerce")
data["Lab. SH"] = pd.to_numeric(data["Lab. SH"], errors="coerce")
data["Eng. AH"] = pd.to_numeric(data["Eng. AH"], errors="coerce")
data["Lab. AH"] = pd.to_numeric(data["Lab. AH"], errors="coerce")

missing_count = data.isna().sum() #Finds the sum of N/A values for each column
missing_percent = (data.isna().mean()*100).round(2) #Finds the percentage of N/A values for each column

distribution = {} #Creates an empty dictionary to store the results for each column.
for col in data.columns: #Loops through every column in the dataset.
    col_series = data[col].dropna() #Removes any missing values

    if col_series.empty:  #If the column has no data left after removing missing values, label it as "No data".
        distribution[col] = "No data"
    elif not is_numeric_dtype(data[col]):
        distribution[col] = "Categorical" #If the column is not numeric, label it as Categorical.
    else:
        s = skew(col_series)#If the column is numeric, calculate its skewness using the skew() function.
        if s >= 1:
            distribution[col] = "Right skewed"  #If skewness is 1 or more, the data has a long tail on the right (a few large values).
        elif s <= -1:
            distribution[col] = "Left skewed" #If skewness is -1 or less, the data has a long tail on the left (a few small values).
        elif abs(s) < 0.5:
            distribution[col] = "normal" #If skewness is close to 0 (between -0.5 and 0.5), the data is fairly symmetrical.
        else:
            distribution[col] = "Slightly skewed"


#Table for % of missing values
summary = pd.DataFrame({
        "Data Type" : data.dtypes.astype(str), #Type of data
        "Missing Count": missing_count, #Total # of missing values per column
        "Missing %": missing_percent,    #Percentage of missing values per column
        "  Distribution": distribution
        }).sort_values("Missing %", ascending=False) #Sorts the values in ascending order

print("\nMISSING VALUES PER ATTRIBUTE + DISTRIBUTION")
print(summary.to_string())

# ########################################################################################################
# #VISUALIZING THE DISTRIBUTION

# # Loop through numeric columns
# for col in data.select_dtypes(include=['int64', 'float64']).columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data[col].dropna(), kde=True, bins=30)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()

# # Loop through categorical columns
# for col in data.select_dtypes(include=['object', 'category']).columns:
#     plt.figure(figsize=(8,4))
#     sns.countplot(x=data[col])
#     plt.title(f"Count Plot of {col}")
#     plt.xticks(rotation=45)
#     plt.show()



# ##################################################################
#     #5) Data Visualization
# ##################################################################

#Plot the frequency of each investigation type
types = [
    "1 - Full Investigation",
    "2 - Full Investigation + Alternate Construction",
    "3 - Alternate Construction",
    "4 - Administrative No Test anticipated (revisions requiring Engineering Review)",
    "5 - Administrative CB review"
]

counts = {t: (data["type_of_investigation"] == t).sum() for t in types}
print(counts)
plt.figure(figsize=(6, 8))
ax = pd.Series(counts).plot(kind="bar", color="skyblue")
total = sum(counts.values())
print("Total:", total)
labels = [f"{c} ({c/total*100:.1f}%)" for c in counts.values()]
ax.bar_label(ax.containers[0], labels=labels)

plt.xlabel("Type of Investigation")
plt.ylabel("Count")
plt.title("Frequency of Each Investigation Type")
plt.title('Average Total Actual Hours by Investigation Type')
plt.legend(['Investigation Type'])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# #Calcuates the average total lab actual hours for each investigation type
avg_lab = (
    data.loc[data["type_of_investigation"].isin(types)]
        .groupby("type_of_investigation")["Lab. AH"]
        .mean()
        .reindex(types)
)

median_lab = (
    data.loc[data["type_of_investigation"].isin(types)]
        .groupby("type_of_investigation")["Lab. AH"]
        .median()
        .reindex(types)
)

summary_lab = pd.DataFrame({"Mean": avg_lab, "Median": median_lab})

ax = summary_lab.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Type of Investigation")
plt.ylabel("Hours")
plt.title("Mean vs Median Lab Actual Hours by Investigation Type")
plt.legend(["Mean", "Median"])
plt.xticks(rotation=30, ha="right")

# add labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.show()

###Calcuates the average total actual eng hours for each investigation type
avg_eng = (
    data.loc[data["type_of_investigation"].isin(types)]
        .groupby("type_of_investigation")["Eng. AH"]
        .mean()
        .reindex(types)
)

median_eng = (
    data.loc[data["type_of_investigation"].isin(types)]
        .groupby("type_of_investigation")["Eng. AH"]
        .median()
        .reindex(types)
)

summary_eng = pd.DataFrame({"Mean": avg_eng, "Median": median_eng})

ax = summary_eng.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Type of Investigation")
plt.ylabel("Hours")
plt.title("Mean vs Median Eng Actual Hours by Investigation Type")
plt.legend(["Mean", "Median"])
plt.xticks(rotation=30, ha="right")

# add labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.show()

# ##################################################################
# #6) correlation between the variables and the targets
# ##################################################################

print("############################### Lab Actual Hours Correlation to Other Attributes #################################")
corr = data.corr(numeric_only=True)
print(corr["Lab. AH"].sort_values(ascending=False))
print("############################### Eng Actual Hours Correlation to Other Attributes #################################")
print(corr["Eng. AH"].sort_values(ascending=False))



# corr = data.corr(numeric_only=True)
# drop_cols = [col for col in corr.columns if "hrs" in col.lower()]
# lab_corr = corr ["Lab. AH"].drop(drop_cols).sort_values(ascending=False)
# eng_corr = corr ["Eng. AH"].drop(drop_cols).sort_values(ascending=False)
# comparison = pd.DataFrame({
#     "Lab. AH": lab_corr,
#     "Eng. AH": eng_corr,
# })
# print(comparison)




# ##################################################################
# #7) Outliers to target
# ##################################################################
