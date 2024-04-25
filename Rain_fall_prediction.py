#!/usr/bin/env python
# coding: utf-8

# 
# # Rainfall prediction

# In[9]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


import warnings
warnings.filterwarnings("ignore")


# 
# ## Laoding the Dataset

# In[2]:


df=pd.read_csv(r"C:\Users\arsha\Downloads\weatherAUS.csv")


# In[77]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# 
# ## clean data

# ### we'll drop all rules that have missing values of the RainToday

# In[6]:


df.drop('RainToday',inplace=True,axis=1)


# In[7]:


df.drop('RainToday',inplace=True,axis=1)


# In[8]:


df.info()


# 
# # exploratory data analysis and visualization

# ### before training a machine learning model , it is always a good idea to explore the distributions of various columns and see how they are related to the target column . Let's explore and visualize the data using the lotly , matplotlib and seaborn libraries.

# In[10]:


df.Location.nunique()


# In[14]:


px.histogram(df, x="Location" , title="Location vs Rainy Days" , color="RainTomorrow")


# In[12]:


px.histogram(df, x="Temp3pm" , title="Temperature at 3 pm vs Rain Tomorrow"  , color="RainTomorrow")


# In[15]:


px.histogram(df, x="RainTomorrow", color="RainTomorrow" , title="Rain Tomorrow vs Rain Today")


# In[17]:


px.scatter(df.sample(2000), x="MinTemp" , y="MaxTemp" , color="RainTomorrow")


# In[18]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='RainTomorrow', y='MaxTemp', data=df)
plt.title('Maximum Temperature vs RainTomorrow')
plt.xlabel('RainTomorrow')
plt.ylabel('MaxTemp')
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(df['MaxTemp'], bins=20, kde=True)
plt.title('Distribution of Maximum Temperature')
plt.xlabel('MaxTemp')
plt.ylabel('Frequency')
plt.show()


# ### The dataset's numerical columns include values such as integers or floats, while the categorical columns contain non-numeric data, such as strings or categorical variables.

# In[22]:


num = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                  'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                  'Temp9am', 'Temp3pm']
cat = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']


# In[23]:


num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())  
])


# In[24]:


cat_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent imputation for missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])


# In[27]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_trans, num),
        ('cat', cat_trans, cat)
    ])


# In[28]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[29]:


pipeline


# In[30]:


x = df.drop('RainTomorrow', axis=1)  # Features
y = df['RainTomorrow'] 
y=y.map({"No":-1,"Yes":1})


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)


# In[74]:


pipeline.fit(x_train,y_train)


# In[33]:


y_pred=pipeline.predict(x_test)


# In[34]:


accuracy=accuracy_score(y_test,y_pred)


# In[35]:


print("accuracy",accuracy)


# In[36]:


pipeline.get_params()


# In[37]:


accuracy=pipeline.score(x_test,y_test)


# In[38]:


accuracy


# In[39]:


query_point = {
    'Date': '2024-04-25',
    'Location': 'Albury',
    'MinTemp': 10.0,
    'MaxTemp': 25.0,
    'Rainfall': 0.0,
    'Evaporation': 5.0,
    'Sunshine': 8.0,
    'WindGustDir': 'NW',
    'WindGustSpeed': 35.0,
    'WindDir9am': 'N',
    'WindDir3pm': 'NNE',
    'WindSpeed9am': 20.0,
    'WindSpeed3pm': 25.0,
    'Humidity9am': 60.0,
    'Humidity3pm': 40.0,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1012.0,
    'Cloud9am': 5.0,
    'Cloud3pm': 3.0,
    'Temp9am': 15.0,
    'Temp3pm': 23.0,
    'RainToday': 'No'
}


# In[40]:


query_df = pd.DataFrame([query_point])


# In[41]:


rainfall_prediction = pipeline.predict(query_df)


# In[42]:


print("Rainfall prediction:", rainfall_prediction)


# ###  Confusion matrix
# 

# In[56]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# ### Classification metrics
# 

# In[57]:


from sklearn.metrics import classification_report

class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)


# ### # Predict probabilities
# 

# In[58]:


y_pred_prob = pipeline.predict_proba(x_test)[:, 1]


# In[59]:


# Adjust threshold
threshold = 0.3
y_pred_adjusted = (y_pred_prob > threshold).astype(int)


# In[76]:


# Updated confusion matrix and classification report

conf_adj = confusion_matrix(y_test, y_pred_adjusted)
class_adj = classification_report(y_test, y_pred_adjusted)

print("\nConfusion Matrix (Adjusted Threshold):")
print(conf_matrix_adjusted)
print("\nClassification Report (Adjusted Threshold):")
print(class_report_adjusted)


# ### # ROC - AUC
# 

# In[61]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[62]:


roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nROC AUC Score:", roc_auc)


# In[63]:


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# ### k-Fold Cross Validation

# In[64]:


from sklearn.model_selection import cross_val_score


# In[72]:


cv_scores = cross_val_score(pipeline, x, y, cv=5)
print("\nCross-Validation Scores:")
print(cv_scores)


# ### # Hyperparameter optimization using GridSearchCV
# 

# In[66]:


param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}


# In[71]:


grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("\nBest Parameters:", grid_search.best_params_)


# # Model evaluation

# In[70]:


best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)
y_pred_best = best_model.predict(x_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("\nAccuracy Score (Best Model):", accuracy_best)


# ### saving the model

# In[55]:


from joblib import dump

# Save the pipeline as a pickle file
dump(pipeline, r'C:\rain_fall_prediction\rainfall_prediction.pkl')


# In[75]:


from joblib import dump

# Save the pipeline as a pickle file
dump(pipeline, r'C:\rain_fall_prediction\rainfall_prediction2.pkl')

