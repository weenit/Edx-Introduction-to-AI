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
zscore_scaler = preprocessing.Normalizer()
data['PlasmaGlucose'] = zscore_scaler.fit_transform(data[['PlasmaGlucose']])
data['DiastolicBloodPressure'] = zscore_scaler.fit_transform(data[['DiastolicBloodPressure']])
data['BMI'] = zscore_scaler.fit_transform(data[['BMI']])

# Normalize Data using min max
minmax_scaler = preprocessing.MinMaxScaler()
data['Pregnancies'] = minmax_scaler.fit_transform(data[['Pregnancies']])
data['TricepsThickness'] = minmax_scaler.fit_transform(data[['TricepsThickness']])
data['SerumInsulin'] = minmax_scaler.fit_transform(data[['SerumInsulin']])
data['DiabetesPedigree'] = minmax_scaler.fit_transform(data[['DiabetesPedigree']])
data['Age'] = minmax_scaler.fit_transform(data[['Age']])

sns.pairplot(data)
plt.show()

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
print(predictions[5:20])
print(y_test[5:20])

# Score Model
print ('Score:', bdt.score(X_test, y_test))

# Evaluate Model
'''How to evaluate model'''
