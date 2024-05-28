#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_df = pd.read_csv(r'D:\christ\Mentorness Machine Learning  Internship\Train_data.csv')
test_df= pd.read_csv(r'D:\christ\Mentorness Machine Learning  Internship\test_data.csv')


# In[3]:


# Size of the training dataset
train_rows, train_cols = train_df.shape
print("Training Dataset Size:", train_rows, "rows,", train_cols, "columns")


# In[4]:


# Size of the test dataset
test_rows, test_cols = test_df.shape
print("Test Dataset Size:", test_rows, "rows,", test_cols, "columns")


# In[5]:


# Display basic information about the training dataset
print("Training Dataset Info:")
print(train_df.info())


# In[6]:


# Columns present in the training dataset
train_columns = train_df.columns
print("\nColumns Present in Training Dataset:")
print(train_columns)


# In[7]:


print("\nTraining Dataset Description:")
print(train_df.describe())


# In[8]:


# Columns present in the test dataset
test_columns = test_df.columns
print("\nColumns Present in Test Dataset:")
print(test_columns)


# In[9]:


# Display basic information about the test dataset
print("\nTest Dataset Info:")
print(test_df.info())


# In[10]:


print("\nTest Dataset Description:")
print(test_df.describe())


# In[11]:


# Number of unique columns in the test dataset
test_unique_cols = test_df.nunique()
print("\nNumber of Unique Columns in Test Dataset:")
print(test_unique_cols)


# In[12]:


# Number of unique columns in the training dataset
train_unique_cols = train_df.nunique()
print("\nNumber of Unique Columns in Training Dataset:")
print(train_unique_cols)


# In[13]:


# Check for missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Separate numerical and non-numerical columns
numerical_cols = train_df.select_dtypes(include=['number']).columns
non_numerical_cols = train_df.select_dtypes(exclude=['number']).columns

# Fill missing values in numerical columns with the mean
train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].mean())
test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].mean())

# For non-numerical columns, you might want to fill missing values with the mode or a specific value
for col in non_numerical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Verify there are no missing values left
print(train_df.isnull().sum())
print(test_df.isnull().sum())


# In[14]:


scaler = StandardScaler()
numerical_features = train_df.columns.difference(['Disease'])

train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])


# In[6]:


X_train = train_df.drop(columns='Disease')
y_train = train_df['Disease']

X_test = test_df.drop(columns='Disease')
y_test = test_df['Disease']


# In[16]:


# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[17]:


# Detailed classification report
print(classification_report(y_test, y_pred))


# In[18]:


# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Use the best model
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_rf_model.predict(X_test)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best, average='weighted')
best_recall = recall_score(y_test, y_pred_best, average='weighted')
best_f1 = f1_score(y_test, y_pred_best, average='weighted')

print(f'Best Accuracy: {best_accuracy}')
print(f'Best Precision: {best_precision}')
print(f'Best Recall: {best_recall}')
print(f'Best F1 Score: {best_f1}')


# In[19]:


# Feature importance
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[3]:


# Plotting the distribution of Cholesterol
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Cholesterol'], kde=True)
plt.title('Distribution of Cholesterol')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('Frequency')
plt.show()


# In[4]:


# Boxplot for Cholesterol
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_df['Cholesterol'])
plt.title('Boxplot of Cholesterol')
plt.xlabel('Cholesterol (mg/dL)')
plt.show()


# In[8]:


# Convert categorical 'Disease' column to numeric
if 'Disease' in train_df.columns:
    train_df['Disease'] = train_df['Disease'].map({'Healthy': 0, 'Diseased': 1})

# Select only numeric columns
numeric_df = train_df.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[4]:


# Bar plot for Disease status
plt.figure(figsize=(10, 6))
sns.countplot(x='Disease', data=train_df)
plt.title('Count of Disease Status')
plt.xlabel('Disease (0: Non-diseased, 1: Diseased)')
plt.ylabel('Count')
plt.show()


# In[5]:


# Scatter plot for Cholesterol vs. BMI
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cholesterol', y='BMI', hue='Disease', data=train_df)
plt.title('Cholesterol vs. BMI')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('BMI')
plt.legend(title='Disease')
plt.show()


# In[6]:


# Violin plot for Cholesterol by Disease status
plt.figure(figsize=(10, 6))
sns.violinplot(x='Disease', y='Cholesterol', data=train_df)
plt.title('Violin plot of Cholesterol by Disease status')
plt.xlabel('Disease (0: Non-diseased, 1: Diseased)')
plt.ylabel('Cholesterol (mg/dL)')
plt.show()


# In[7]:


# Joint plot for Cholesterol and BMI
sns.jointplot(x='Cholesterol', y='BMI', data=train_df, kind='scatter', hue='Disease')
plt.show()

