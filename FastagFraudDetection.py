#!/usr/bin/env python
# coding: utf-8

# In[183]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


# In[169]:


df = pd.read_csv('FastagFraudDetection.csv')
df.head()


# In[170]:


df.info()


# In[171]:


df.describe()


# In[172]:


sns.countplot(x='Fraud_indicator', data=df)
plt.show()


# In[173]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[ ]:





# In[174]:


# Create a new feature for the difference between Transaction_Amount and Amount_paid
df['Amount_diff'] = df['Transaction_Amount'] - df['Amount_paid']

# Create a new feature for the ratio between Transaction_Amount and Amount_paid
df['Amount_ratio'] = df['Transaction_Amount'] / df['Amount_paid']

# Create a new feature for the transaction time
df['Transaction_time'] = pd.to_datetime(df['Timestamp']).dt.time

# Create a new feature for the transaction day of the week
df['Transaction_day'] = pd.to_datetime(df['Timestamp']).dt.dayofweek

# Create a new feature for the transaction hour of the day
df['Transaction_hour'] = pd.to_datetime(df['Timestamp']).dt.hour

# Drop unnecessary columns
df.drop(['Timestamp'], axis=1, inplace=True)


# In[175]:


X = df.drop('Fraud_indicator', axis=1)
y = df['Fraud_indicator']


# In[176]:


categorical_cols = X.select_dtypes(include=['object']).columns


# In[177]:


one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = one_hot_encoder.fit_transform(X[categorical_cols])


# In[178]:


feature_names = one_hot_encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=feature_names)


# In[179]:


X.drop(categorical_cols, axis=1, inplace=True)


# In[180]:


X = pd.concat([X, X_encoded_df], axis=1)


# In[181]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[182]:


X_train.replace([np.inf, -np.inf], 999999, inplace=True)
X_test.replace([np.inf, -np.inf], 999999, inplace=True)

# Replace very large values with a large finite number
X_train.clip(lower=-999999, upper=999999, inplace=True)
X_test.clip(lower=-999999, upper=999999, inplace=True)


# In[157]:


X_train.dropna(inplace=True)
X_test.dropna(inplace=True)


# In[185]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:





# In[186]:


class_labels = ['Fraud', 'Not Fraud']
if 'Fraud' not in y_test.unique():
    class_labels = ['Not Fraud', 'Fraud']
y_test_mapped = ['Fraud' if i == class_labels[1] else 'Not Fraud' for i in y_test]
y_pred_mapped = ['Fraud' if i == class_labels[1] else 'Not Fraud' for i in y_pred]


# In[ ]:





# In[187]:


accuracy = accuracy_score(y_test_mapped, y_pred_mapped)
precision = precision_score(y_test_mapped, y_pred_mapped, pos_label=class_labels[1])
recall = recall_score(y_test_mapped, y_pred_mapped, pos_label=class_labels[1])
f1 = f1_score(y_test_mapped, y_pred_mapped, pos_label=class_labels[1])


# In[188]:


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 score: {f1:.2f}')


# In[ ]:




