#import needed modules
from datetime import datetime
from pandas_datareader import data
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ticker list
symbList = ['NEE', 'D', 'DUK', 'SO', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES', 'ED', 'PEG', 'FE', 'AWK', 'EIX', 'DTE', 'PPL', 'ETR', 'AEE', 'CMS', 'EVRG', 'ATO', 'LNT', 'NI', 'PNW', 'AES', 'CNP', 'NRG']

start_date = '2015/01/01'
end_date = datetime.now()

def get_data(symbList=symbList,start_date=start_date,end_date=end_date):

    df_list = []
    for i in symbList:
        print('Loading data: ',i)
        dt = data.DataReader(i, "yahoo", start=start_date, end=end_date)
        dt.rename(columns={'Adj Close': 'price'}, inplace=True)
        df_list.append(dt)



    min_date = max(df.dropna().index[0] for df in df_list)
    max_date = min(df.dropna().index[-1] for df in df_list)

    s_list = []
    for df in df_list:
        s = df['price']
        s_list.append(s)
    df =  pd.concat(s_list, axis=1, keys=symbList).dropna()
    print(df.head())
    return df


def stats(x,y):
    # run Odinary Least Squares regression to find hedge ratio
    # and then create spread series
    df1 = pd.DataFrame({'x':x,'y':y})
    est = sm.OLS(df1.y, df1.x)
    est = est.fit()
    df1['hr'] = est.params[0]
    hr= df1['hr'].mean()
    hrr = round(hr)
    df1['spread'] = df1.y - (df1.x * hrr)

    mean = df1.spread.mean()
    std = df1.spread.std()

    cadf = ts.adfuller(df1.spread)

    # Run OLS regression on spread series and lagged version of itself
    spread_lag = df1.spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = df1.spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)

    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()

    halflife = round(-np.log(2) / res.params[1], 0)

    return [hrr,df1.hr.std(),mean,std,halflife,cadf,hrr-hr,df1]


def plot1(x,y,i):
    plt.plot(y,label=i[1])
    plt.plot(x,label=i[0])
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot2(x,y,i):
    sns.jointplot(y, x ,color='b')
    plt.title(i)
    plt.show()



def plot3(df1, mean,std,i, r0 = -3,r1 = 4):
    plt.plot(df1.spread)

    for x in range (r0,r1):
        plt.axhline(y=mean + (std * x) , color='b', linestyle='-')

    plt.axhline(y=mean, color='r', linestyle='-')
    plt.title(i)
    plt.show()





df =get_data()

combs = list(combinations(symbList,2))
print(len(combs))
print(combs)

ind = 1

for i in combs:
    print('##########')
    print(ind,'/',len(combs))
    ind += 1

    print(i)

    x = df[i[0]]
    y = df[i[1]]
    st = stats(x,y)



    if abs(st[5][0]) > abs(st[5][4]['5%']) and st[4]>0:


        print('mean : ',st[2],' std : ',st[3])
        print('Hedge Ratio mean : ',st[0],' std : ',st[1],' round : ',st[-2])
        print ('Augmented Dickey Fuller test statistic =',st[5][0])
        print ('Augmented Dickey Fuller p-value =',st[5][1])
        print ('Augmented Dickey Fuller 1%, 5% and 10% test statistics =',st[5][4])


        print  ('Halflife = ',st[4])

        #plot1(x=x,y=y,i=i)
        plot2(x=x, y=y, i=i)
        plot3(st[-1],st[2],st[3],i)

    else:
        print('Failed Test')
    #breakpoint()

