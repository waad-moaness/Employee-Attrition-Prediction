# HR Employee Attrition Prediction

## 1. Problem Statement

Employee turnover (attrition) is a major cost for organizations, leading to loss of institutional knowledge, hiring expenses, and reduced team morale. This project focuses on predicting **which employees are at risk of leaving** the company.

Unlike a simple descriptive analysis, this project treats the problem as a **Binary Classification task**. By predicting the probability of attrition (`Yes` vs. `No`) based on demographics, job role, and satisfaction metrics, HR departments can proactively identify at-risk talent and intervene to improve retention.

## 2. Dataset Description

The project utilizes the **IBM HR Analytics Employee Attrition & Performance** dataset, which consists of demographic and job-related data for approximately 1,500 employees.

* **Source:** [Kaggle: IBM HR Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Type:** Structured Tabular Data.
* **Target Variable:** `Attrition` (Binary: "Yes" / "No").
* **Input Features:**
    * **Numerical:** `Age`, `MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`, `DailyRate`, etc.
    * **Categorical:** `Department`, `JobRole`, `MaritalStatus`, `OverTime`, `BusinessTravel`, etc.

## 3. EDA Summary

Exploratory Data Analysis (EDA) revealed specific characteristics of the data that influenced the modeling strategy:

* **Target Imbalance:** The dataset is heavily imbalanced, with only **~16%** of employees labeled as "Attrition = Yes".
    * *Strategy:* Used `stratify` during splitting and applied `scale_pos_weight` (in XGBoost) and `class_weight='balanced'` (in Logistic Regression/Random Forest) to penalize false negatives.
* **Overfitting Risk:** The dataset is relatively small (~1,470 rows). Complex models like Deep Trees easily achieved 100% training accuracy but failed to generalize.
    * *Strategy:* Focused on shallow trees (`max_depth=3`) and strict regularization.

## 4. Modeling Approach & Metrics

### Modeling Approach
The project followed a robust Machine Learning pipeline using `scikit-learn` and `Logistic Regression`:

1.  **Preprocessing:**
    * **Categorical Features:** Handled using `OneHotEncoder` (ignoring unknown categories).
    * **Numerical Features:** Scaled using `StandardScaler` (crucial for Logistic Regression).
    * **Pipeline:** Preprocessing and Modeling steps were bundled into a single `sklearn.pipeline.Pipeline` object to prevent data leakage and simplify deployment.
2.  **Model Selection:** Three models were trained and tuned:
    * **Logistic Regression:** Established as a strong baseline due to the small dataset size.
    * **Random Forest:** Tuned for `min_samples_leaf` to prevent overfitting.
    * **XGBoost Classifier:** Tuned for depth and learning rate.
    * **Logistic Regression regularized:** Final model hsa the highest accuracy and the least overfitting.

### Metrics
The models were evaluated using **ROC AUC (Area Under the Curve)** to measure how well they distinguish between "Stay" and "Leave" classes, regardless of the decision threshold.

| Model | Validation AUC | Notes |
| :--- | :--- | :--- |
| Logistic Regression | 0.8046 | Very strong baseline; highly interpretable. |
| Random Forest (Tuned) | 0.8045 | Suffered from overfitting before tuning leaf size. |
| XGBoost (Tuned) | 0.8122 | good performance. Achieved by constraining `max_depth` to 3.but suffered from overfitting |
| **Logistic Regression (regularized)** | **00.8154** | **Best performance.** Achieved by constraining `regularization parameter` to 0.01. |

### Final Model

- **Logistic Regression:**  
  - Regularization `C=0.01`  
  - `class_weight='balanced'`  
  - Solver: `"liblinear"`  
  - **Training AUC:** 0.8614  
  - **Validation AUC:** 0.8154  

> Logistic Regression was chosen as the final model due to its strong generalization on the validation set and simplicity, making it interpretable and robust.

---

## 5. How to Run Locally and Via Docker

### Prerequisites
* Python 3.9+
* Docker (optional but recommended)
* Pipenv (for dependency management)

### Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/waad-moaness/Employee-Attrition-Prediction.git
    cd Employee-Attrition-Prediction
    ```
2.  **Prerequisites**
    You must have Python and Pipenv installed.
    ```bash
    pip install pipenv
    ```

3.  **Install Dependencies:**
    Use `sync` to install the exact versions from the lock file.
    ```bash
    pipenv sync
    ```

4.  **Activate the Virtual Environment:**
    ```bash
    pipenv shell
    ```

5.  **Run the Prediction Service:**
    ```bash
    python predict.py
    ```
    *The service will start on `http://0.0.0.0:9696`*

### Running Via Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t employee-attrition-prediction .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 9696:9696 employee-attrition-prediction
    ```

## 6. API Usage Example

The model expects a JSON payload containing the employee's details.

**Endpoint:** `POST /predict`

**Example Request Body:**
```json
{
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

```
## 7. Known Limitations / Next Steps

### Limitations

* **Small Dataset:** With only ~1,470 records, complex models like Gradient Boosting are prone to overfitting. The winning model had to be severely constrained (shallow trees) to work effectively.
* **Static Snapshot:** The data represents a single snapshot in time. It does not capture how an employee's satisfaction or performance changes over time (time-series aspect).

### Next Steps

* **Feature Engineering:** Create interaction features (e.g., `Income / Age` ratio) or binning for `Age` to capture non-linear generational trends.
* **Threshold Tuning:** Instead of the default 0.5 threshold, optimize the decision threshold to recall more "At Risk" employees (minimizing False Negatives), as losing talent is expensive.
* **Cloud Deployment:** Deploy the containerized service to AWS Lambda or Google Cloud Run for serverless scalability.* **Threshold Tuning:** Instead of the default 0.5 threshold, optimize the decision threshold to recall more "At Risk" employees (minimizing False Negatives), as losing talent is expensive.
* **Cloud Deployment:** Deploy the containerized service to AWS Lambda or Google Cloud Run for serverless scalability.


## Demo Video


https://github.com/user-attachments/assets/ee5b29c3-17a7-479d-814d-e99fbf57eccb


