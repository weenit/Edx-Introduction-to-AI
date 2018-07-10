import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# get data
file_path = 'Blood donation data.csv'
data = pd.read_csv(file_path)
sns.pairplot(data)
plt.show()

# select columns in dataset
features = pd.DataFrame(data.iloc[:,[0,1,2,3]].values)

head_title = ['Recency', 'Frequency', 'Monetary', 'Time']
features.columns = head_title
features.head()

# scale: zscore normalization
zscore_scaler = preprocessing.Normalizer()
features['Recency'] = zscore_scaler.fit_transform(features[['Recency']])
features['Frequency'] = zscore_scaler.fit_transform(features[['Frequency']])
features['Monetary'] = zscore_scaler.fit_transform(features[['Monetary']])
features['Time'] = zscore_scaler.fit_transform(features[['Time']])

sns.pairplot(data)
plt.show()

# K-Mean Clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
