GitHub Repository Description
California House Price Prediction API 🏠📊

A machine learning-powered REST API built with FastAPI to predict California house prices using a trained Random Forest Regressor. The API supports both individual house price predictions through JSON requests and batch predictions by uploading CSV files.

--> Features
-->House Price Prediction using a trained Random Forest model
--> FastAPI REST API with automatic API documentation
--> Prediction using 8 California housing features:
Median Income
House Age
Average Rooms
Average Bedrooms
Population
Average Occupancy
Latitude
Longitude

-->CSV Batch Prediction — upload a CSV file and receive predictions as a downloadable CSV
--> Pydantic validation for incoming prediction data
--> Health Check Endpoint to verify API and model status
-->Structured error handling for invalid requests and prediction failures
-->Converts model predictions into estimated USD house prices
--> API Endpoints

Method	Endpoint	Description
GET	/	API welcome and status information
GET	/health	Check API and model health
POST	/predict	Predict the price of a single house
POST	/predictFileData	Upload a CSV and generate batch predictions

--> Tech Stack
Python
FastAPI
Scikit-learn
Random Forest Regressor
Pandas
NumPy
Pydantic
Joblib
Uvicorn

--Model Files
The project uses two serialized files:

houseModel.joblib
houseFeatures.joblib

houseModel.joblib contains the trained machine learning model, while houseFeatures.joblib stores the model's feature information.


--Project Goal
The goal of this project is to demonstrate how a trained machine learning model can be deployed as a production-style REST API, allowing users or applications to obtain house price predictions programmatically.

--Example Prediction
A request can provide house information such as:

{
  "MedInc": 8.3,
  "HouseAge": 41,
  "AvgRoom": 6.98,
  "AvgBedroom": 1.02,
  "population": 322,
  "AvgOcuup": 2.55,
  "Lattitude": 37.88,
  "Longitude": -122.23
}

The API returns an estimated house price along with a guidance range.

Short GitHub "About" Description
If you mean the short description shown beside your repository name, I'd use:

Machine learning-powered FastAPI for California house price prediction with single and CSV batch prediction support. 🏠📊

And suggested GitHub topics:

python machine-learning fastapi scikit-learn random-forest rest-api pandas house-price-prediction data-science ml-api regression uvicorn

One small recommendation before you publish: don't upload any sensitive credentials, .env files, virtual environments, or unnecessary large model artifacts to GitHub. A .gitignore is worth adding.
