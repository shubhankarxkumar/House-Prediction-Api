from sklearn.datasets import fetch_california_housing #dataset for practice
import pandas as pd #to work with tabular data 

data = fetch_california_housing() #well to use that dataset we first need to store it somewhere to access it

#it is to understand the data and its structure

df = pd.DataFrame(data.data, columns=data.feature_names)

df["Price"] = data.target

print("Shape", df.shape) #tell me the the number of rows and column
print(df.head()) # to return first 5 rows
print(df.describe()) #tell about the data strc
