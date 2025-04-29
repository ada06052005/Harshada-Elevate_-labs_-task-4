# Harshada-Elevate_-labs_-task-4
Project Overview
This project aims to build a predictive model for classifying breast cancer tumors as malignant (M) or benign (B) using a dataset of various tumor characteristics. We use a Logistic Regression model for binary classification, and evaluate the model's performance using metrics like precision, recall, F1-score, and ROC AUC.

Dataset
The dataset used in this project is the well-known Breast Cancer Wisconsin (Diagnostic) dataset. It contains measurements of 30 features extracted from breast cancer cell images, such as radius, texture, perimeter, area, smoothness, and compactness. The target variable is diagnosis, where:

M represents malignant tumors (coded as 1)

B represents benign tumors (coded as 0)

Dataset Columns
id: Unique identifier for each tumor

diagnosis: Tumor classification (M or B)

radius_mean, texture_mean, etc.: Mean measurements of various tumor features

Unnamed: 32: A column with no data (removed from analysis)

Project Steps
1. Importing Libraries
We use the following libraries for data manipulation, visualization, and model building:

python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
2. Data Preprocessing
Loading Data: The dataset is loaded from a CSV file.

Data Cleaning: The id and Unnamed: 32 columns are dropped. The target variable diagnosis is encoded as binary (1 for malignant, 0 for benign).

3. Data Exploration
Checking for missing values and basic statistics.

Visualizing target class distribution to see the balance between malignant and benign classes.

4. Feature and Target Split
The features (X) are separated from the target (y), where X contains the tumor characteristics, and y is the target variable (diagnosis).

5. Train-Test Split
We split the dataset into a training set (80%) and a testing set (20%) using train_test_split.

6. Feature Scaling
Standard scaling is applied to the features using StandardScaler to improve model performance.

7. Model Training (Logistic Regression)
A Logistic Regression model is trained on the scaled training data.

python
Copy
Edit
model = LogisticRegression()
model.fit(X_train, y_train)
8. Model Evaluation
Classification Report: Shows precision, recall, F1-score, and accuracy for both classes.

ROC AUC Score: The model's performance is evaluated using the Area Under the Curve (AUC) score.

Confusion Matrix: A heatmap of the confusion matrix is visualized to show the true positives, false positives, true negatives, and false negatives.

9. ROC Curve
The ROC curve is plotted to visualize the trade-off between sensitivity and specificity.

Results
Classification Report
plaintext
Copy
Edit
              precision    recall  f1-score   support
           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
ROC AUC Score: 0.997
Confusion Matrix:
A heatmap of the confusion matrix is displayed to assess the model's performance visually.

ROC Curve:
A plot of the ROC curve is generated to visualize the true positive rate vs. false positive rate.

Conclusion
The Logistic Regression model performs exceptionally well with a high accuracy of 97% and a very high ROC AUC score of 0.997. The model achieves a good balance between precision and recall, especially for malignant tumors.

Requirements
Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Installation
You can install the required libraries using pip:
pip install pandas numpy matplotlib seaborn scikit-learn

