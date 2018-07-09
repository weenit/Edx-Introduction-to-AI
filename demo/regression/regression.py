''' Use linear regression to predict calories '''
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read data
exercise = pd.read_csv('exercise.csv')
calories = pd.read_csv('calories.csv')

data = pd.merge(exercise, calories, on='User_ID')
# print(data.head())

# Convert Gender to categorial
enc = preprocessing.LabelEncoder()
enc.fit(['male', 'female'])
data['Gender_Categorial'] = pd.DataFrame(enc.transform(data['Gender']))
# data['Gender_Categorial_Inv'] = pd.DataFrame(enc.inverse_transform(data['Gender_Categorial']))

# Create features
data['Duration_Sqr'] = data['Duration'] ** 2
data['Heart_Rate_Sqr'] = data['Heart_Rate'] ** 2

'''
    apply math operation
    calculate log calories to ensure no negative prediction
'''

# scale value: zscore normalization
zscore_scaler = preprocessing.Normalizer()
data['Height'] = zscore_scaler.fit_transform(data[['Height']])
data['Weight'] = zscore_scaler.fit_transform(data[['Weight']])
data['Heart_Rate'] = zscore_scaler.fit_transform(data[['Heart_Rate']])
data['Heart_Rate_Sqr'] = zscore_scaler.fit_transform(data[['Heart_Rate_Sqr']])
data['Body_Temp'] = zscore_scaler.fit_transform(data[['Body_Temp']])

# scale value: MinMax [Age, Duration, Duration_Sqr]
min_max_scaler = preprocessing.MinMaxScaler()
data['Age'] = min_max_scaler.fit_transform(data[['Age']])
data['Duration'] = min_max_scaler.fit_transform(data[['Duration']])
data['Duration_Sqr'] = min_max_scaler.fit_transform(data[['Duration_Sqr']])

# print(data.head())

''' Remove unwanted features before training '''
features = data
del features['User_ID']
del features['Gender']
y = features.Calories
del features['Calories']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

# print(predictions[0:5])

'''Plot predictions vs actual value'''
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

print ('Score:', model.score(X_test, y_test))



