import pickle
import numpy as np
import pandas as pd  
from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


app = FastAPI(title = 'employee_attrition_prediction')


class EmployeeFeatures(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    OverTime: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Attrition Prediction API. Go to /docs to see the API."}


def normalize_input(df):
    for col in ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']:
        df[col] = df[col].str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
    return df

@app.post('/predict')
def predict_attrition(emp: EmployeeFeatures):
    if pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    data = pd.DataFrame([emp.dict()])  
    data = normalize_input(data)     
    proba = float(pipeline.predict_proba(data)[:, 1][0])
    pred = pipeline.predict(data)[0]
    return {
        "attrition_probability": round(proba, 3),
        "attrition": "Yes" if pred == 1 else "No"
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)