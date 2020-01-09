import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime
#import math 
import matplotlib.pyplot as plt
import random
#from itertools import accumulate
from tkinter import *
import seaborn as sns
from pandas_datareader.data import DataReader
from regression_output import rd2 as rd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.pyplot import figure as Figure

class MyGUI(tk.Tk):
    def __init__(self,master):
        self.fig=Figure(figsize=(6,10),dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig,master)

        self.toolbar=NavigationToolbar2Tk(self.canvas,master)
        #self.toolbar.update()
        self.canvas._tkcanvas.pack(padx=2, pady=2)

#Functions
def VWAP(data):
    return(np.cumsum(data.Bought*data.Close)/
    np.cumsum(data.Bought)).round(2)

def ploting():
    
    conversion()

    df2 = rd[(rd['Brand'] == conversion.brand) & (rd['Size (group)'] == conversion.size) & (rd['UHD Segment'] == conversion.segment) & (rd['Date'] >= conversion.start_date) & (rd['Date'] <= conversion.end_date) & (rd['Outlet'] == conversion.outlet) & (rd['Sub Res'] == conversion.resolution)]
    
    df2_rollingmean = df2['Units'].rolling(window=4).mean()
    df2_rollingstd = df2['Units'].rolling(window=4).mean()
    
    
    df3 = df2[np.abs(df2.Productivity-df2.Productivity.mean())<=(3*df2.Productivity.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'
    X = df3[['Productivity','Price']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    X = X.loc[:,X.isnull().mean() < .5]
    X = X[~np.isnan(X).any(axis=1)]
    Y = X['Productivity']
    X = X.drop('Productivity', axis=1)
    
    

    graphs = tk.Frame()

    first = plt.subplot2grid((14,5), (0, 0), rowspan=3, colspan=8)
    first.plot(df2['Date'], df2["Units"], label='Units')
#     first.plot(df2_rollingmean, color='red', label='Rolling Mean')
    plt.title('{} {} {} {} ({})  [{} - {}]'.format(e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get())) 
    plt.legend(loc='best', numpoints = 1, prop={'size':7})
    
    top = plt.subplot2grid((14,5), (4, 0), rowspan=3, colspan=8)
    top.plot(df2['Date'], df2["Price"], label='Price', color='blue')
    plt.legend(loc='best', numpoints = 1, prop={'size':7})
    
    
    bottom = plt.subplot2grid((14,5), (8,0), rowspan=2, colspan=8)
    bottom.bar(df2['Week'], df2['PriceDownFlag'])
    plt.title('PriceDown Flag')
    plt.gcf().set_size_inches(10,10)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    y_pred2 = regressor.predict(X_test)
    
    bottom2 = plt.subplot2grid((14,5), (11,0), rowspan=4, colspan=3)
    bottom2.scatter(X, Y)
    plt.plot(X, Y_pred, color='blue')
    plt.plot(X_test, y_pred2, color='green', linestyle="dotted")
    plt.ylabel('ST/Loc')
    plt.xlabel('Price')
#     plt.show()
    
    

#     root.mainloop()

def conversion():
#     print("Data conversion")
    conversion.outlet = e1.get()
    conversion.brand = e2.get()
    conversion.size = e3.get()
    conversion.segment = e4.get()
    conversion.resolution = e5.get()
    conversion.start_date = datetime.datetime.strptime(e6.get(),'%Y-%m-%d')
    conversion.end_date = datetime.datetime.strptime(e7.get(),'%Y-%m-%d')

#-----------------------------------------------------------------------------

def close_window (): 
     master.destroy()

master = Tk()

Label(master, text="Outlet").grid(row=0)
Label(master, text="Brand").grid(row=1)
Label(master, text="Size").grid(row=2)
Label(master, text="Segment").grid(row=3)
Label(master, text="Resolution").grid(row=4)
Label(master, text="Start Date").grid(row=5)
Label(master, text="End Date").grid(row=6)

e1 = Entry(master)
e1.insert(10,"WMT001")
e2 = Entry(master)
e2.insert(10,"VIZIO")
e3 = Entry(master)
e3.insert(10,"46 - 52")
e4 = Entry(master)
e4.insert(10,"UHD ENTRY")
e5 = Entry(master)
e5.insert(10,"UHD")
e6 = Entry(master)
e6.insert(10,"2019-01-01")
e7 = Entry(master)
e7.insert(10,"2019-10-29")

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)

Button(master, text='Cancel', command=close_window).grid(row=7, column=0, sticky=W, pady=4)

Button(master, text='Plot', command=ploting).grid(row=7, column=2, sticky=W, pady=4)

#df2 = web.DataReader(conversion.moniker, 'yahoo', conversion.start_date, conversion.end_date)
#df2 = web.DataReader(conversion.moniker, 'yahoo', datetime.datetime(2010,1,1),datetime.datetime(2014,1,1) )

master.mainloop()