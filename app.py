# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BuildingEnergy import BuildingEnergy
import numpy as np
import joblib
import pandas as pd
# 2. Create the app object
app = FastAPI()
# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# mapping: user input name -> training column
feature_map = {
    'Relative_Compactness': 'X1',
    'Surface_Area': 'X2',
    'Wall_Area': 'X3',
    'Roof_Area': 'X4',
    'Overall_Height': 'X5',
    'Orientation': 'X6',
    'Glazing_Area': 'X7',
    'Glazing_Area_Distribution': 'X8'
}

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post("/predict")
def predict_building_consumption(data: BuildingEnergy):
    # Convert Pydantic object to dict
    input_dict = data.dict()
    
    # Map real names to training column names
    mapped_input = {feature_map[k]: v for k, v in input_dict.items()}
    
    # Convert to DataFrame with the same order as training
    input_df = pd.DataFrame([mapped_input])[['X1','X2','X3','X4','X5','X6','X7','X8']]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    return {"Heating Load": prediction.tolist()[0][0],
            "Cooling Load": prediction.tolist()[0][1]}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
    