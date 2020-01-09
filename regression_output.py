import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
# from etl import df5,rd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
# from mlxtend.plotting import plot_linear_regression
from sklearn.model_selection import train_test_split
from numpy import inf


from matplotlib.pyplot import figure


df = pd.read_csv('NPD.csv', sep = ',', engine='python')

df['Date'] = pd.to_datetime(df['Date'])

rd = pd.DataFrame(df, columns =['Date','Outlet', 'Brand','SKU','Size (group)', 'UHD Segment','Sub Res', 'Year', 'Week', 'Units', 'Stores','Price', 'Productivity'])


rd2 = df.groupby(['Date','Outlet', 'Brand','Size (group)','UHD Segment', 'Sub Res', 'Year', 'Week'])['Units', 'Dollars', 'Stores'].sum()
# print(rd2.info())

rd2['Price'] = rd2['Dollars']/rd2['Units']


# rd2 = rd2.sort(rd2['Date'])
rd2['PriceChange'] = rd2['Price']-rd2['Price'].shift(1)

rd2 = rd2.reset_index()
rd2['Price'][rd2['Price'] == inf] = np.nan
rd2['Productivity'] = rd2['Units']/rd2['Stores']
rd2.dropna(axis=0, how='any', inplace=True)


# print(rd2['Price'])

rd['Units'] = rd['Units'].astype(int)
rd['Date'] = pd.to_datetime(rd['Date'])
rd['PriceChange'] = rd['Price']-rd['Price'].shift(1)

# print(rd)

def Flag(x):
    if x < 0:
        return 1
    return 0



rd['PriceDownFlag'] = rd['PriceChange'].map(lambda x: Flag(x))
rd2['PriceDownFlag'] = rd2['PriceChange'].map(lambda x: Flag(x))
# print(rd['PriceDownFlag'])
print(rd2[['Price', 'PriceChange', 'PriceDownFlag']])
sns.countplot(data=rd, x='PriceDownFlag')
# rd['Price'] = rd['NPD ASP']
rd = rd[(rd['Units'] > 200) & (rd['Stores'] > 2)]
# rd = rd[(rd['PriceChange'] < -.01) & (rd['Lift'] > 0) & (rd['Lift'] < 1.5) & (rd['PriceChange'] > -1)]

rd.head()

# rd = rd[(rd['Date'] != '2018-11-19') & (rd['Date'] != '2018-11-12') & (rd['Date'] != '2018-11-26')]
df = rd

# brand = input("Brand: ")
brand = "LG"
# size = input("Size: ")
size = '65'
# segment = input("Segment: ")
segment = "UHD PREMIUM"
# outlet = input("Outlet: ")
outlet = "Remaining Retail"

df = df[(df['Brand'] == brand) & (df['Size (group)'] == size) & (df['UHD Segment'] == segment) & (df['Year'] >= 2019) & (df['Outlet'] == outlet)]
df2 = df
countplot = sns.countplot(data=df, x='PriceDownFlag')
print(countplot)
stat1 = df['Productivity'].describe()

df['Productivity'].describe()

df = df[np.abs(df.Productivity-df.Productivity.mean())<=(2*df.Productivity.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'

X = df[['Productivity','Price']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
X = X.loc[:,X.isnull().mean() < .5]
X = X[~np.isnan(X).any(axis=1)]
Y = X['Productivity']
X = X.drop('Productivity', axis=1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='blue')
# plt.ylabel('ST/Loc')
# plt.xlabel('Price')
# plt.show()
# plt.savefig('sample.png')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred2 = regressor.predict(X_test)

# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='blue')
# plt.plot(X_test, y_pred2, color='green')
# plt.ylabel('ST/Loc')
# plt.xlabel('Price')
# plt.show()

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
# print(est2.summary())

# Stats 2

stat2 = est2.summary()


lm = LinearRegression()
lm.fit(X,Y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X.reset_index(drop=True)))
MSE = (sum((Y-predictions)**2))/(len(newX)-len(newX.columns))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]

p_values = np.delete(p_values,0,0)

# print(pd.DataFrame(np.transpose(lm.coef_)))

coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)
coefficients.columns = ['variable', 'coefficient']

# print(pd.DataFrame({'intercept':[lm.intercept_]}))                         

reg = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose([lm.intercept_]))], axis = 1)
reg.columns = ['variable', 'intercept']

pvalues = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(p_values))], axis = 1)
pvalues.columns = ['variable', 'pvalue']

m1 = pd.merge(coefficients, pvalues)
m1
union = pd.merge(reg, m1)
result = union.fillna(0)

# print(result)


                     



