import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json


df = pd.read_csv('NPD.csv', sep = ',', engine='python')

rd = pd.DataFrame(df, columns =['Date','Outlet', 'Brand','SKU','Size (group)', 'UHD Segment','Sub Res', 'Year', 'Week', 'Units', 'Stores','Price', 'Productivity'])
rd['Units'] = rd['Units'].astype(int)
rd['Date'] = pd.to_datetime(rd['Date'])
rd = rd[(rd['Units'] > 200) & (rd['Stores'] > 50)]

df = rd
brand = 'SAMSUNG'
size = "46 - 52"
segment = "UHD ENTRY"
outlet = "WMT001"

df = df[(df['Brand'] == brand) & (df['Size (group)'] == size) & (df['UHD Segment'] == segment) & (df['Year'] >= 2019) & (df['Outlet'] == outlet)]

# print(df.head())
df = df[np.abs(df.Productivity-df.Productivity.mean())<=(3*df.Productivity.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'

X = df[['Productivity','Price']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
X = X.loc[:,X.isnull().mean() < .5]
X = X[~np.isnan(X).any(axis=1)]
y = X['Productivity']
X = X.drop('Productivity', axis=1)


# print(y.describe())
# print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[350]]))




