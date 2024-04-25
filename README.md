# Rainfall Prediction using Logistic Regression

## Overview
In this project, the goal is to predict whether it will rain or not the next day in Australia based on various weather observations. The dataset contains about 10 years of daily weather data from multiple weather stations across Australia.

## Dataset
- [Click Here]https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/download?datasetVersionNumber=2 to access the dataset.
- The dataset contains the following features:
    - Date
    - Location
    - MinTemp
    - MaxTemp
    - Rainfall
    - Evaporation
    - Sunshine
    - WindGustDir
    - WindGustSpeed
    - WindDir9am
    - WindDir3pm
    - WindSpeed9am
    - WindSpeed3pm
    - Humidity9am
    - Humidity3pm
    - Pressure9am
    - Pressure3pm
    - Cloud9am
    - Cloud3pm
    - Temp9am
    - Temp3pm
    - RainToday
    - RISK_MM
    - RainTomorrow

## Steps
1. **Import libraries**: Import necessary libraries for data manipulation, visualization, and model building.
2. **Import dataset**: Load the dataset into the environment.
3. **Exploratory data analysis**: Analyze the dataset to understand its structure, distribution, and relationships.
4. **Declare feature vector and target variable**: Define the input features (X) and the target variable (y).
5. **Split data into separate training and test set**: Divide the dataset into training and testing sets.
6. **Feature engineering**: Perform feature engineering tasks such as handling missing values, encoding categorical variables, etc.
7. **Feature scaling**: Scale the features to ensure they have the same range.
8. **Model training**: Train a Logistic Regression model using the training data.
9. **Predict results**: Use the trained model to make predictions on the test data.
10. **Check accuracy score**: Evaluate the model's performance using accuracy score.
11. **Confusion matrix**: Analyze the confusion matrix to understand the model's performance further.
12. **Classification metrics**: Calculate various classification metrics such as precision, recall, F1-score, etc.
13. **Adjusting the threshold level**: Adjust the classification threshold to improve model performance.
14. **ROC - AUC**: Plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC) score.
15. **k-Fold Cross Validation**: Perform k-Fold Cross Validation to assess model stability and generalization.
16. **Hyperparameter optimization using GridSearchCV**: Fine-tune the model's hyperparameters using GridSearchCV.
17. **Model evaluation**: Evaluate the final model on various evaluation metrics.
18. **Results and conclusion**: Summarize the results and draw conclusions from the analysis.

## Streamlit App Deployment
The trained Logistic Regression model has been deployed on a Streamlit app for easy interaction. Users can input weather details, and the app predicts whether it will rain or not the next day.
