#import needed modules
from datetime import datetime
from pandas_datareader import data
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns



#choose ticker pairs for our testing
symbList = ['SPY','XLU']

start_date = '2000/01/01'
end_date = datetime.now()

#download data from Yahoo Finance
y=data.DataReader(symbList[0], "yahoo", start=start_date, end=end_date)
x=data.DataReader(symbList[1], "yahoo", start=start_date, end=end_date)

#rename column to make it easier to work with later
y.rename(columns={'Adj Close':'price'}, inplace=True)
x.rename(columns={'Adj Close':'price'}, inplace=True)

#make sure DataFrames are the same length
min_date = max(df.dropna().index[0] for df in [y, x])
max_date = min(df.dropna().index[-1] for df in [y, x])

y = y[(y.index>= min_date) & (y.index <= max_date)]
x = x[(x.index >= min_date) & (x.index <= max_date)]

#print(x)


plt.plot(y.price,label=symbList[0])
plt.plot(x.price,label=symbList[1])
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.jointplot(y.price, x.price ,color='b')
plt.show()

#run Odinary Least Squares regression to find hedge ratio
#and then create spread series
df1 = pd.DataFrame({'y':y['price'],'x':x['price']})
est = sm.OLS(df1.y,df1.x)
est = est.fit()
df1['hr'] = -est.params[0]
df1['spread'] = df1.y - (df1.x * -df1.hr)

mean = df1.spread.mean()
std = df1.spread.std()

plt.plot(df1.spread)

for x in range (-3,4):
    plt.axhline(y=mean + (std * x) , color='b', linestyle='-')

plt.axhline(y=mean, color='r', linestyle='-')
plt.show()

cadf = ts.adfuller(df1.spread)
print('mean : ',mean,' std : ',std)
print('Hedge Ratio mean : ',-df1.hr.mean(),' std : ',df1.hr.std())
print ('Augmented Dickey Fuller test statistic =',cadf[0])
print ('Augmented Dickey Fuller p-value =',cadf[1])
print ('Augmented Dickey Fuller 1%, 5% and 10% test statistics =',cadf[4])



#Run OLS regression on spread series and lagged version of itself
spread_lag = df1.spread.shift(1)
spread_lag.iloc[0] = spread_lag.iloc[1]
spread_ret = df1.spread - spread_lag
spread_ret.iloc[0] = spread_ret.iloc[1]
spread_lag2 = sm.add_constant(spread_lag)

model = sm.OLS(spread_ret,spread_lag2)
res = model.fit()


halflife = round(-np.log(2) / res.params[1],0)

print  ('Halflife = ',halflife)

