import joblib
import io
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("houseModel.joblib")
feature = joblib.load("houseFeatures.joblib")


# -------------------------
# Input schema
# -------------------------

class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median Income")
    HouseAge: float = Field(gt=0, description="Avg age of the Block")
    AvgRoom: float = Field(gt=0, description="Avg room in the Block")
    AvgBedroom: float = Field(gt=0, description="Avg Bedroom in the Block")
    population: float = Field(gt=0, description="Population in that Block area")
    AvgOcuup: float = Field(gt=0, description="Avg Occupancy")
    Lattitude: float = Field(ge=32, le=42, description="Latitude of the Property")
    Longitude: float = Field(
        ge=-125,
        le=-114,
        description="Longitude of the Property"
    )


# -------------------------
# Home
# -------------------------

@app.get("/")
def home():
    return {
        "message": "California House Prediction Model",
        "status": "Active, Good to Go",
        "endpoint": "Send the POST request to get the Pricing"
    }


# -------------------------
# Health
# -------------------------

@app.get("/health")
def health():
    return {
        "Message": "Prediction Model Health",
        "Status": "Running",
        "Model": "RandomForestRegressor",
        "Features": feature
    }


# -------------------------
# Single prediction
# -------------------------

@app.post("/predict")
def predict(house: HouseFeatures):

    try:

        inputData = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AvgRoom,
            "AveBedrms": house.AvgBedroom,
            "Population": house.population,
            "AveOccup": house.AvgOcuup,
            "Latitude": house.Lattitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(inputData)[0]

        priceUsd = predicted * 100000

        return {
            "Predicted Price": f"${priceUsd:,.0f}",
            "Predicted Price Shorted by": f"${predicted:,.2f} Hundred Thousand Dollars",
            "Guidance range": f"${priceUsd - 39000:,.0f} to ${priceUsd + 39000:,.0f}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction of the model failed",
                "reason": f"Probable reason: {str(e)}"
            }
        )


# -------------------------
# CSV prediction
# -------------------------

@app.post("/predictFileData")
async def predictFile(file: UploadFile = File(...)):

    # Check file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "Error": "Uploaded file is not a CSV file",
                "Fix": "Please upload a CSV file"
            }
        )

    # Read uploaded file
    contents = await file.read()

    # Convert CSV into DataFrame
    df = pd.read_csv(io.BytesIO(contents))

    # Model's exact feature names
    requiredColumns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    # Check missing columns
    missingColumns = [
        col for col in requiredColumns
        if col not in df.columns
    ]

    if missingColumns:
        raise HTTPException(
            status_code=400,
            detail={
                "Message": "Some required columns are missing",
                "Missing Columns": missingColumns,
                "Required Columns": requiredColumns
            }
        )

    # Check empty file
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )

    try:

        # Prediction
        predictions = model.predict(df[requiredColumns])

        # Convert model output to USD
        predictionsUSD = predictions * 100000

        # Add prediction column
        df["PredictedPriceUSD"] = predictionsUSD

        # Format price
        df["PredictedPriceUSD"] = df["PredictedPriceUSD"].apply(
            lambda x: f"${x:,.0f}"
        )

        # Convert DataFrame back to CSV
        output = df.to_csv(index=False)

        # Return downloadable CSV
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed",
                "reason": str(e)
            }
        )