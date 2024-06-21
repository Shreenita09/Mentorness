#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify


# In[2]:


df = pd.read_csv('Salary Prediction of Data Professions.csv')


# In[3]:


df.head()


# # EDA

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


plt.figure(figsize=(10, 6))
sns.countplot(x='DESIGNATION', data=df)
plt.title('Distribution of Job Roles')
plt.show()


# In[8]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='AGE', y='SALARY', data=df)
plt.title('Relationship between Age and Salary')
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='UNIT', y='SALARY', data=df)
plt.title('Salary Distribution by Business Unit')
plt.show()


# # FEATURE ENGG

# In[11]:


df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
df['DOJ'] = pd.to_datetime(df['DOJ'])

df['EXPERIENCE'] = df['PAST EXP'] + (df['CURRENT DATE'] - df['DOJ']).dt.days / 365
df['RATINGS_CAT'] = pd.cut(df['RATINGS'], bins=[0, 2, 4, 6, 8, 10], labels=[1, 2, 3, 4, 5])


# # DATA PREPROCESSING

# In[12]:


le = LabelEncoder()
df['SEX'] = le.fit_transform(df['SEX'])
df['DESIGNATION'] = le.fit_transform(df['DESIGNATION'])
df['UNIT'] = le.fit_transform(df['UNIT'])


# In[13]:


X = df.drop(['SALARY', 'FIRST NAME', 'LAST NAME'], axis=1)
y = df['SALARY']


# In[14]:


X['CURRENT DATE'] = (X['CURRENT DATE'] - pd.to_datetime('1970-01-01')).dt.days
X['DOJ'] = (X['DOJ'] - pd.to_datetime('1970-01-01')).dt.days

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

models = [
    Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LinearRegression())
    ]),
    Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', DecisionTreeRegressor())
    ]),
    Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestRegressor(n_estimators=100))
    ]),
    Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', GradientBoostingRegressor())
    ])
]

scores = []
for model in models:
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores.append(-np.mean(score))  # Note the negative sign because of 'neg_mean_squared_error' scoring

best_model_index = np.argmax(scores)
best_model = models[best_model_index]

print(f'Best model: {best_model.steps[1][1].__class__.__name__}')


# In[16]:


results = []
for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append((model.__class__.__name__, mae, mse, rmse, r2))


# In[17]:


print('Model Evaluation Results:')
print('-------------------------')
for result in results:
    print(f'Model: {result[0]}')
    print(f'MAE: {result[1]:.2f}')
    print(f'MSE: {result[2]:.2f}')
    print(f'RMSE: {result[3]:.2f}')
    print(f'R2 Score: {result[4]:.2f}')
    print('-------------------------')


# In[18]:


app = Flask(__name__)

# Define the scaler and models variables
scaler = StandardScaler()
models = [...]  # Define the models list

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_new = pd.DataFrame(data, index=[0])
        X_new_scaled = scaler.transform(X_new)
        y_pred = models[2].predict(X_new_scaled)[0]  # Use the best-performing model (Random Forest)
        return jsonify({'predicted_salary': y_pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:


print('Recommendations:')
print('--------------')
print('1. Experience is a significant factor in determining salaries.')
print('2. Job roles with higher ratings tend to have higher salaries.')
print('3. Business units with higher average salaries tend to have more experienced professionals.')
print('4. Improving performance ratings and gaining more experience can lead to higher salaries.')


# In[ ]:




