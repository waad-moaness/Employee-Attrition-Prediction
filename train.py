import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv",engine='python')

    df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)

    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

    categorical = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'Gender',
        'JobRole',
        'MaritalStatus',
        'OverTime'
    ]

    numerical = [
        'Age',
        'DailyRate',
        'DistanceFromHome',
        'Education',
        'EnvironmentSatisfaction',
        'HourlyRate',
        'JobInvolvement',
        'JobLevel',
        'JobSatisfaction',
        'MonthlyIncome',
        'MonthlyRate',
        'NumCompaniesWorked',
        'PercentSalaryHike',
        'PerformanceRating',
        'RelationshipSatisfaction',
        'StockOptionLevel',
        'TotalWorkingYears',
        'TrainingTimesLastYear',
        'WorkLifeBalance',
        'YearsAtCompany',
        'YearsInCurrentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]


    for col in categorical: 
        df[col] = df[col].str.lower().str.replace(' ' ,'_' ).str.replace(r'[^\w\s]', '', regex=True)


    X = df.drop('Attrition', axis= 1)
    y = df.Attrition

    return X , y , numerical , categorical

def train_model(X, y, numerical, categorical):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            C=0.01,
            solver="liblinear",
            class_weight="balanced",
            random_state=1,
            max_iter=1000
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline , output_file):

    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)



X_train , y_train ,numerical , categorical = load_data()
pipeline = train_model(X_train , y_train ,numerical , categorical)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')