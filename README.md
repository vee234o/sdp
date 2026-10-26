# Sleep Disorder Prediction System

## Project Overview
This project is an end-to-end machine learning pipeline and interactive web application designed to predict whether an individual is at risk of sleep disorders (Insomnia, Sleep Apnea, or None) based on lifestyle habits and clinical vitals.

## Deployed Application
Access the live web application here: [https://sdpredictor.streamlit.app](https://sdpredictor.streamlit.app/)

## Dataset
The model is trained on the Sleep Health and Lifestyle Dataset. It contains demographic details, sleep metrics, physical activity levels, BMI categories, blood pressure, heart rate, and sleep disorder classifications.

## Features Used
* Gender
* Age
* Sleep Duration
* Quality of Sleep
* Physical Activity Level
* Stress Level
* BMI Category
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Heart Rate
* Daily Steps

## Data Pipeline and Cleaning
1. Dropped unique row identifiers that carry no predictive value.
2. Handled missing values in the sleep disorder column by assigning them to the `None` category.
3. Split composite blood pressure strings into separate systolic and diastolic numerical features.
4. Consolidated inconsistent category labels in the BMI category column.
5. Encoded categorical features into numerical values for model compatibility.

## Model Training and Evaluation
* Split the dataset into training and testing sets with stratification.
* Standardized continuous features using `StandardScaler`.
* Trained Random Forest and Gradient Boosting classifiers.
* Evaluated model performance using accuracy score, weighted F1 score, classification reports, and confusion matrix heatmaps.

## Installation and Setup
Clone the repository and install the required dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App
Ensure your trained model pickle file (`sleep_disorder_rf_model.pkl`) and scaler pickle file (`feature_scaler.pkl`) are in the project root directory, then run the application:

```bash
streamlit run app.py
```
