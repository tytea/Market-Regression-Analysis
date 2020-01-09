import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import inf
from decimal import Decimal

sns.set(rc={'figure.figsize': (18, 6)})

df = pd.read_csv('NPD.csv', sep = ',', engine='python')

# print(df.head())

rd = pd.DataFrame(df, columns =['Date','Outlet', 'Brand','Model','Size (group)', 'UHD Segment','Sub Res', 'Year', 'Week', 'Units', 'Stores', 'Lift','Price % Change', 'ProductivityLift', 'Dollars'])

rd['Units'] = rd['Units'].astype(int)
rd['Date'] = pd.to_datetime(rd['Date'])
rd['PriceChange'] = rd['Price % Change']
rd = rd[(rd['Units'] > 100) & (rd['Stores'] > 2)]
rd = rd[(rd['PriceChange'] < -.01) & (rd['Lift'] > 0) & (rd['Lift'] < 1.5) & (rd['PriceChange'] > -1)]

print(rd)


df2 = pd.DataFrame(df, columns =['Date','Outlet', 'BrandGroup', 'Brand','Model', 'UnitsFilter','Size (group)', 'UHD Segment','Sub Res', 'Year', 'Week', 'PriceBand', 'Units', 'Stores', 'Dollars'])
df2['Units'] = df2['Units'].astype(int)
df2['Date'] = pd.to_datetime(df2['Date'])

df3 = df2.groupby(['Outlet', 'BrandGroup', 'Brand','Date','Year', 'Week', 'Size (group)','UHD Segment', 'Sub Res', 'UnitsFilter'])['Units', 'Dollars','Stores'].sum()
df3 = df3.reset_index()



#  Weeknum Function
def weeknum(week):
    if week < 10:
        return '0' +  str(week)
    else:
        return str(week)
    
#  YearMonth Function
def yearmonth(month):
    if month < 10:
        return '0' +  str(month)
    else:
        return str(month) 
    
    
    
df3['YearWk'] = df['Year'].astype(str) + df['Week'].apply(weeknum)
df3['YearWk'] = df3['YearWk'].astype(int)
df3['UnitsPerStore'] = df3['Units']/df3['Stores']
df3['ASP'] = df3['Dollars']/df3['Units']
df3['ASP'].replace(np.nan, 0, inplace=True)
# df3['Size'] = df3['Size'].astype(int)
df3['Units'] = df3['Units'].astype(int)
df3['Units'].replace(0, np.nan, inplace=True)

df3.dropna(axis=0, how='any', inplace=True)
df3['ASP'] = df3['ASP'].astype(int)

# print(df3.head())


rretail = df3

vizio_rr = df3[df3['Brand'] == 'VIZIO']


df4 = df3[df3['Year'] >= 2018]

df4['Month'] = df4['Date'].map(lambda x: x.month)

# df4['Month'] = df4['Date'].map(lambda x: x.strftime('%m'))

# df['YearMonth'] = df['ArrivalDate'].map(lambda x: 100*x.year + x.month)
# df4 = df4[df4['Size'] == 50]
# df4 = df4[df4['Sub Res'] == 'UHD']


VZ = df4[df4['BrandGroup'] == 'VIZIO']
T1 = df4[df4['BrandGroup'] == 'T1']
TCL = df4[df4['BrandGroup'] == 'TCL']

# print(df4.head())
df5 = df4.groupby(['Outlet', 'BrandGroup', 'Brand','Year', 'Month', 'Size (group)','UHD Segment', 'Sub Res', 'UnitsFilter'])['Units', 'Dollars','Stores'].sum()
df5 = df5.reset_index()
df5['UnitsPerStore'] = df5['Units']/df5['Stores']
df5['ASP'] = df5['Dollars']/df5['Units']
df5['ASP'].replace(np.nan, 0, inplace=True)
# df5['Size'] = df5['Size'].astype(int)
df5['Units'] = df5['Units'].astype(int)
df5['Units'].replace(0, np.nan, inplace=True)
df5['YearMonth'] = df5['Year'].astype(str) + df['Week'].apply(yearmonth)

df5.dropna(axis=0, how='any', inplace=True)
df5['ASP'] = df5['ASP'].astype(int)

df5['UnitsPerStore'][df5['UnitsPerStore'] == -inf] = 0

# print(df5.head())
