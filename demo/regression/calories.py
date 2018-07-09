import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data from csv
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# join dataframes on by User_ID
data = pd.merge(exercise, calories, on='User_ID')
# print(data.head())

# Create a scatter plot matrix to visualize relationship between data
sns.pairplot(data, size=2)
plt.show()
