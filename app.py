import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px


# Load the saved pipeline
pipeline = load(r"C:\rain_fall_prediction\rainfall_prediction2.pkl")

# Function to make predictions
def predict_rainfall(query_point):
    # Prepare the query point as a DataFrame
    query_df = pd.DataFrame([query_point])
    # Predict rainfall
    rainfall_prediction = pipeline.predict(query_df)
    return rainfall_prediction

st.image(r"C:\Users\arsha\Downloads\rain fall image.jpg")


# Define the query point inputs using Streamlit widgets
st.title('Rainfall Predictor')

st.sidebar.header('Provide Weather Details')

date = st.sidebar.date_input("Date", value=pd.to_datetime('today'))
location = st.sidebar.text_input("Location")
min_temp = st.sidebar.number_input("MinTemp", value=1.0)
max_temp = st.sidebar.number_input("MaxTemp", value=0.0)
rainfall = st.sidebar.number_input("Rainfall", value=0.0)
evaporation = st.sidebar.number_input("Evaporation", value=0.0)
sunshine = st.sidebar.number_input("Sunshine", value=0.0)
wind_gust_dir = st.sidebar.selectbox("WindGustDir", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
wind_gust_speed = st.sidebar.number_input("WindGustSpeed", value=0.0)
wind_dir_9am = st.sidebar.selectbox("WindDir9am", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
wind_dir_3pm = st.sidebar.selectbox("WindDir3pm", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
wind_speed_9am = st.sidebar.number_input("WindSpeed9am", value=0.0)
wind_speed_3pm = st.sidebar.number_input("WindSpeed3pm", value=0.0)
humidity_9am = st.sidebar.number_input("Humidity9am", value=0.0)
humidity_3pm = st.sidebar.number_input("Humidity3pm", value=0.0)
pressure_9am = st.sidebar.number_input("Pressure9am", value=0.0)
pressure_3pm = st.sidebar.number_input("Pressure3pm", value=0.0)
cloud_9am = st.sidebar.number_input("Cloud9am", value=0.0)
cloud_3pm = st.sidebar.number_input("Cloud3pm", value=0.0)
temp_9am = st.sidebar.number_input("Temp9am", value=0.0)
temp_3pm = st.sidebar.number_input("Temp3pm", value=0.0)
rain_today = st.sidebar.selectbox("RainToday", ['Yes', 'No'])

# Prepare the query point dictionary
query_point = {
    'Date': date,
    'Location': location,
    'MinTemp': min_temp,
    'MaxTemp': max_temp,
    'Rainfall': rainfall,
    'Evaporation': evaporation,
    'Sunshine': sunshine,
    'WindGustDir': wind_gust_dir,
    'WindGustSpeed': wind_gust_speed,
    'WindDir9am': wind_dir_9am,
    'WindDir3pm': wind_dir_3pm,
    'WindSpeed9am': wind_speed_9am,
    'WindSpeed3pm': wind_speed_3pm,
    'Humidity9am': humidity_9am,
    'Humidity3pm': humidity_3pm,
    'Pressure9am': pressure_9am,
    'Pressure3pm': pressure_3pm,
    'Cloud9am': cloud_9am,
    'Cloud3pm': cloud_3pm,
    'Temp9am': temp_9am,
    'Temp3pm': temp_3pm,
    'RainToday': rain_today
}

# Make prediction
if st.sidebar.button('Predict Rainfall'):
    prediction = predict_rainfall(query_point)
    if prediction[0] == 1:
        st.success("🌧️ Rainfall prediction: Yes")
    else:
        st.success("☀️ Rainfall prediction: No")

# Additional features
df=pd.read_csv(r"C:\Users\arsha\Downloads\weatherAUS.csv")
st.header('Visualizing Data Insights')

st.subheader('Location vs Rainy Days')
fig1 = px.histogram(df, x="Location", title="Location vs Rainy Days", color="RainTomorrow")
st.plotly_chart(fig1)

st.subheader('Rain Tomorrow vs Rain Today')
fig2 = px.histogram(df, x="RainTomorrow", color="RainTomorrow", title="Rain Tomorrow vs Rain Today")
st.plotly_chart(fig2)

st.subheader('Min Temperature vs Max Temperature')
fig3 = px.scatter(df.sample(2000), x="MinTemp", y="MaxTemp", color="RainTomorrow")
st.plotly_chart(fig3)




