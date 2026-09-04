from sklearn.datasets import fetch_california_housing #sklearm.datasets has many practice dataset from which we are using one for our proj
from sklearn.ensemble import RandomForestRegressor #well same model diff toll used for prediction of number
from sklearn.model_selection import train_test_split # usd to train and test our model
from sklearn.metrics import r2_score, mean_absolute_error # metrics to measure our model r2_score - model rate, mean - how far we are far from ans
import pandas as pd # to work with dataset
import joblib 

data = fetch_california_housing() # load the dataset to use in our program

#data.data - to go inside that dataaset and bring all the raw data(num)
#data.feature_name - to list down all the name of the column
#stored in x -  by convection in ml it means it is the data we will feed to our model

x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target #it will go inside our scikitlearn lib and fetch all the house prices and will store it inside x

print(f"total data:{x.shape[0]}")

xtrain , xtest, ytrain, ytest = train_test_split(x , y, test_size=0.2, random_state=42)


#train model
model = RandomForestRegressor(
  n_estimators=100,
  random_state=42
)

model.fit(xtrain, ytrain)

yPread = model.predict(xtest)
mae = mean_absolute_error(ytest, yPread)

r2 = r2_score(ytest, yPread)

print(f"average error ${mae *100000:,.0f}")
print(r2)

joblib.dump(model, "houseModel.joblib")
joblib.dump(list(x.columns),"houseFeatures.joblib")