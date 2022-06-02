import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from data import data

app = FastAPI()

with open('model1', 'rb') as file:
    model = pickle.load(file)


@app.post('/predict')
def predict(df: data):
    df = df.dict()
    print(df)
    ph_2 = df['ph_2']**2
    sulf_2 = df['sulf_2']**2
    chl_2 = df['chl_2']**2
    Solids = df['Solids']
    Hardness = df['Hardness']
    Organic_carbon = df['Organic_carbon']
    Conductivity = df['Conductivity']
    Trihalomethanes = df['Trihalomethanes']
    Turbidity = df['Turbidity']
    prediction = model.predict([[ph_2, sulf_2, chl_2, Solids, Hardness, Organic_carbon, Conductivity, Trihalomethanes, Turbidity]])
    if prediction == 1:
        ans = 'safe water'
    else: 
        ans = 'unsafe water'
    return {
        'ans': ans
    }

if __name__ == '__main__':
    uvicorn.run(app ,host='localhost', port=5000)
