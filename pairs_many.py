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
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from time import sleep
from numpy import cumsum, polyfit, log, sqrt, std, subtract,var, log10

pd.set_option('display.max_columns', 500)
# ticker list

#XLU
# symbList = ['NEE', 'D', 'DUK', 'SO', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES', 'ED', 'PEG', 'FE', 'AWK', 'EIX', 'DTE', 'PPL', 'ETR', 'AEE', 'CMS', 'EVRG', 'ATO', 'LNT', 'NI', 'PNW', 'AES', 'CNP', 'NRG']

#XLE
symbList =['CVX', 'XOM', 'COP', 'PSX', 'EOG', 'VLO', 'WMB', 'SLB', 'KMI', 'MPC', 'PXD', 'OXY', 'OKE', 'HES', 'CXO', 'BKR', 'HAL', 'COG', 'FANG', 'NOV', 'HFC', 'MRO', 'DVN', 'APA', 'NBL', 'FTI', 'HP']

#XLRE
# symbList =['AMT', 'CCI', 'PLD', 'EQIX', 'DLR', 'SBAC', 'PSA', 'AVB', 'EQR', 'SPG', 'WELL', 'ARE', 'O', 'ESS', 'WY', 'BXP', 'CBRE', 'MAA', 'PEAK', 'DRE', 'VTR', 'EXR', 'UDR', 'HST', 'REG', 'IRM', 'VNO', 'FRT', 'AIV', 'KIM', 'SLG']

#XLC
#symbList =['FB', 'GOOGL', 'GOOG', 'TMUS', 'EA', 'NFLX', 'CHTR', 'ATVI', 'DIS', 'VZ', 'CMCSA', 'T', 'TWTR', 'TTWO', 'OMC', 'CTL', 'FOXA', 'VIAC', 'DISCK', 'IPG', 'DISH', 'LYV', 'FOX', 'NWSA', 'DISCA', 'NWS']

#XLY
# symbList =['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'TGT', 'DG', 'ROST', 'EBAY', 'ORLY', 'GM', 'YUM', 'AZO', 'MAR', 'CMG', 'HLT', 'F', 'DLTR', 'VFC', 'APTV', 'BBY', 'DHI', 'LVS', 'LEN', 'TIF', 'ULTA', 'TSCO', 'KMX', 'GRMN', 'GPC', 'NVR', 'EXPE', 'DRI', 'HAS', 'AAP', 'WYNN', 'MGM', 'LKQ', 'WHR', 'PHM', 'RCL', 'CCL', 'BWA', 'MHK', 'NWL', 'LEG', 'TPR', 'RL', 'HBI', 'PVH', 'HOG', 'HRB', 'NCLH', 'KSS', 'LB', 'CPRI', 'JWN', 'UAA', 'GPS']

#XLP
# symbList = ['PG', 'PEP', 'KO', 'WMT', 'MDLZ', 'MO', 'COST', 'PM', 'CL', 'KMB', 'EL', 'GIS', 'WBA', 'STZ', 'SYY', 'KR', 'CLX', 'MNST', 'HSY', 'ADM', 'MKC',  'TSN', 'CHD', 'CAG', 'K', 'SJM', 'HRL',  'CPB',  'TAP', 'COTY']

# symbList = ['NVDA','SOXX','AMD','INTC','TSM','QQQ','XLK']

# symbList = ['O','ACC','AWP','CCI','BRX','CDR','KIM','UBP','PK','MGP','HT','STAY','VICI','XHR','CXW','GEO']

# symbList = ['GS','BAC','WFC','BK','BLK','BX','JPM','C','AXP','CME','CBOE']

start_date = '2016/01/01'
end_date = '2020/08/01'

#end_date = datetime.now()

lags = range(2,100)
def hurst_ernie_chan(p):

    variancetau = []; tau = []

    for lag in lags:

        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = subtract(p[lag:], p[:-lag])
        variancetau.append(var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = polyfit(log10(tau),log10(variancetau),1)

    hurst = m[0] / 2

    return hurst

def get_data(symbList=symbList,start_date=start_date,end_date=end_date):

    df_list = []
    for i in symbList:
        print('Loading data: ',i)


        try: dt = data.DataReader(i, "yahoo", start=start_date, end=end_date)
        except:
            print('Loading data Failed:',i)
            continue
        sleep(0.5)
        dt.rename(columns={'Close': 'price'}, inplace=True)
        df_list.append(dt)



    min_date = max(df.dropna().index[0] for df in df_list)
    max_date = min(df.dropna().index[-1] for df in df_list)

    s_list = []
    for df in df_list:

        log = False
        if log == True:
            s = np.log(df['price'])
        else:
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

    dw = durbin_watson(est.resid)
    jb = jarque_bera(est.resid)
    #print(dw)
    df1['hr'] = est.params[0]
    hr= df1['hr'].mean()
    hrr = round(hr)
    #print(hrr)
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

    df1['next_s'] = df1.spread.shift(-1)
    df1['cross'] = (
            ((df1.spread >= mean) & (df1.next_s < mean)) |
            ((df1.next_s > mean) & (df1.spread <= mean)) |
            (df1.spread == mean))

    cross = df1.cross.sum()


    # Coint test
    result = ts.coint(df1['x'], df1['y'])

    # Hurst exponent

    hurst = hurst_ernie_chan(df1['spread'])




    return [hrr,df1.hr.std(),mean,std,halflife,cadf,hrr-hr,df1,dw,est,jb,cross,result,hurst]


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

dict_list = []
for i in combs:
    print('##########')
    print(ind,'/',len(combs))
    ind += 1

    print(i)

    x = df[i[0]]
    y = df[i[1]]
    st = stats(x,y)

    #plot2(x=x, y=y, i=i)

    if abs(st[12][0]) >= abs(st[12][2][1]):# and abs(st[5][0]) > abs(st[5][4]['5%']):# and st[4]>0  and st[0] >0 and st[4] < 50 and abs(st[12][0]) >= abs(st[12][2][0]) and st[12][0] < 0.5:#and st[5][1] < (0.1/len(combs)) and st[8]>1.5 and st[8] < 2.5 :

        print(st[9].summary())


        print('mean : ',st[2],' std : ',st[3])
        print('Hedge Ratio mean : ',st[0],' std : ',st[1],' round : ',st[6])
        print ('Augmented Dickey Fuller test statistic =',st[5][0])
        print ('Augmented Dickey Fuller p-value =',st[5][1])
        print ('Augmented Dickey Fuller 1%, 5% and 10% test statistics =',st[5][4])
        print('Cointegration test statistic =', st[12][0])
        print('Cointegration test p-value  =', st[12][1])
        print('Cointegration test 1%, 5% and 10% test statistics =', st[12][2])
        print('Hurst Exponent:', st[13])

        print  ('Halflife = ',st[4])

        print('Mean crossings = ', st[11])

        #plot1(x=x,y=y,i=i)
        plot2(x=x, y=y, i=i)
        plot3(st[7],st[2],st[3],i)

        dic = {'pair':i,'mean':st[2],'std':st[3],'adf_test':st[5][0],'adf_p_value':st[5][1],'hedge_ratio_round':st[0],'hedge_ratio_error':st[6],'halflife':st[4],'Durbin':st[8],'crosses': st[11],'coint':st[12][0],'coint-pvalue':st[12][1],'Hurst Exp':st[13] }
        dict_list.append(dic)

    else:
        print('Failed Test')
    #breakpoint()

df = pd.DataFrame(dict_list).sort_values(by=['Hurst Exp'], ascending=False)

# df.to_csv('/Users/joanarau-schweizer/PycharmProjects/pairs_trading/DB/pairs.csv')

print(df)
print(len(df.index))

