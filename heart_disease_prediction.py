import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# 1. Load the dataset
# The dataset is sourced from the UCI Machine Learning Repository.
url = 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
df = pd.read_csv(url)

# -- FIX: Handle the categorical 'thal' column before EDA and modeling --
# The 'thal' column contains object types that need to be converted to a numerical format.
# We will use one-hot encoding to represent the different categories.
# The original dataset from UCI has a '?' value for some rows, which this CSV represents as '?.
# We will replace '?' with the most frequent value (mode).
if 'thal' in df.columns:
    # See if there are non-standard values like '?'
    if df['thal'].dtype == 'object':
        # Replace non-standard markers with the mode
        mode_val = df['thal'].mode()[0]
        # In this dataset, some versions have '?' or other placeholders.
        # This line handles them, though in the current tensorflow-hosted CSV, it may not be needed.
        df['thal'] = df['thal'].replace('?', mode_val)
        
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=['thal'], drop_first=True)

# 2. Exploratory Data Analysis (EDA)
print("First 5 rows of the dataset after encoding 'thal':")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nChecking for missing values:")
print(df.isnull().sum())

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Distribution of Heart Disease (1 = Disease, 0 = No Disease)')
plt.show()

# Visualize correlation matrix
plt.figure(figsize=(14, 10))
# Now that 'thal' is numeric, this will work.
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# 3. Data Preparation
# The target column is named 'target'
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Decision Tree Classification Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve and AUC Score
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f'AUC Score: {auc_score:.4f}')

# 6. Feature Importance Analysis
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance in Predicting Heart Disease')
plt.show()

print("\nTop important features:")
print(feature_importance_df) 