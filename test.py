
import pandas as pd
import requests

url = 'http://localhost:9696/predict'

test = {
    "Age": 28,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 866,
    "Department": "Sales",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Medical",
    "EnvironmentSatisfaction": 4,
    "Gender": "Male",
    "HourlyRate": 84,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 1,
    "MaritalStatus": "Single",
    "MonthlyIncome": 8463,
    "MonthlyRate": 23490,
    "NumCompaniesWorked": 0,
    "OverTime": "No",
    "PercentSalaryHike": 18,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 4,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 6,
    "TrainingTimesLastYear": 4,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
}

response = requests.post(url, json=test)

predictions = response.json()

print(f'Is the employee likely to leave the company: {predictions['prediction']}')