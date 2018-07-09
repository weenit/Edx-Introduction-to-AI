import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecissionTreeClasifier

# Import data
diabetes_filepath = 'diabetes.csv'
doctors_filepath = 'doctors.csv'

diabetes = pd.read_csv(diabetes_filepath)
doctors = pd.read_csv(doctors_filepath, encoding = 'ISO-8859-1')

# Join data: Using Patient_ID
data = pd.merge(diabetes, doctors, on = 'PatientID')
sns.pairplot(data)
plt.show()

# Math Operation: for age (log of age)
data['Age'] = np.log(data['Age'])

# Normalize Data using zscore

# Normalize Data using min max

# Edit metadata for training
features = data
del features['PatientID']
del features['Physician']
y = features['Diabetic']
del features['Diabetic']

# Split the data 70-30
X_train, X_test, y_train, y_test = train_test_split(features, y, size = 0.3)

# Two-Class Boosted Decision Tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

# Train Model
bdt.fit(X_train, y_train)

# Predict
predictions = bdt.predict(X_test)
# Score Model

# Evaluate Model
