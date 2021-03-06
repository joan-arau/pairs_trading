# coding: utf-8
# ##################################################################
# Pair Trading adapted to backtrader
# with PD.OLS and info for StatsModel.API
# author: Remi Roche
##################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)



import matplotlib


from itertools import combinations

from math import log
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import datetime


# The above could be sent to an independent module
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind


import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 30, 20  # that's default image size for this interactive session
pylab.rcParams['font.family'] = 'sans-serif'
pylab.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
pylab.rcParams['font.serif'] = ['Bitstream Vera Sans']
pylab.rcParams["font.size"] = "10"

# from obs import Spread

import pandas as pd
from backtrader.indicators.basicops import PeriodN
import statsmodels.api as sm
### TODO ###
#
# Implement multiple stocks
#
# Delta hedge portfolio using /MES
#
###
class spread_class(PeriodN):
    '''
    Calculates a regression of data1 on data0 using ``pandas.ols``

    Uses ``pandas``
    '''
    _mindatas = 2  # ensure at least 2 data feeds are passed

    packages = (
        ('pandas', 'pd'),
    )

    lines = ('beta','spread')
    params = (('period', 100),)

    def next(self):
        y, x = (pd.Series(d.get(size=self.p.period)) for d in self.datas)
        #r_beta = sm.ols(y=y, x=x, window_type='full_sample')

        model_Simple = sm.OLS(y, x).fit()
        r_beta = model_Simple.params
        #print(r_beta)
        self.lines.beta[0] = r_beta
        self.lines.spread[0] = self.data0[0] - (self.lines.beta[0]*self.data1[0])

class PairTradingStrategy(bt.Strategy):
    params = dict(
        period=252,
        stake=10,
        qty1=0,
        qty2=0,
        printout=True,
        upper=1,
        lower=-1,
        up_medium=0.1,
        low_medium=-0.1,
        stop_up = 2,
        stop_down = -2,
        status=0,
        portfolio_value=10000,
    )

    def log(self, txt, dt=None):
        if self.p.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f' % order.executed.price
                self.log(buytxt, order.executed.dt)
            else:
                selltxt = 'SELL COMPLETE, %.2f' % order.executed.price
                self.log(selltxt, order.executed.dt)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass  # Simply log

        # Allow new orders
        self.orderid = None

    def __init__(self):


        # To control operation entries
        self.orderid = None
        self.qty1 = self.p.qty1
        self.qty2 = self.p.qty2
        self.upper_limit = self.p.upper
        self.lower_limit = self.p.lower
        self.up_medium = self.p.up_medium
        self.low_medium = self.p.low_medium
        self.stop_up =  self.p.stop_up
        self.stop_down = self.p.stop_down

        # self.status = self.p.status
        self.portfolio_value = self.p.portfolio_value

        self.plts = {}



        self.data_dic = {}

        for i in list(enumerate(self.datas))[1:]:

            self.data_dic[i[1]._name] = i[1]

        self.combs = list(combinations(self.data_dic.keys(), 2))
        print(len(self.combs))
        print(self.combs)

        self.bench = list(enumerate(self.datas))[0][1]
        print(self.bench._name)

        self.combs_stats = []
        self.pos_traker = {}
        for a, b in self.combs:
            self.data0 = self.data_dic[a]
            self.data1 = self.data_dic[b]



            # Signals performed with PD.OLS :
            self.transform = btind.OLS_TransformationN(self.data0, self.data1,
                                                       period=self.p.period,
                                                       plot = True,
                                                       plotname = str(a+'-'+b+" Spread"),
                                                       subplot = True,
                                                       plotabove= True,
                                                       # plotmaster = self.bench
                                                       )
            self.zscore = self.transform.zscore

            self.beta =  btind.OLS_BetaN(self.data0, self.data1,
                                                       period=self.p.period,plot = False)

            # self.spread = spread_class(self.data0, self.data1, period=self.p.period)

            self.coint = btind.CointN(self.data0, self.data1,
                                        period=self.p.period,plot = False)

            self.combs_stats.append([[a,b],self.transform,self.zscore,self.beta,self.coint])

            self.pos_traker[str(a+'/'+b)] = 0
            self.plts[str(a+'/'+b)] = []




            # Checking signals built with StatsModel.API :
            # self.ols_transfo = btind.OLS_Transformation(self.data0, self.data1,
            #                                             period=self.p.period,
            #                                             plot=True)

    def next(self):

        if len(self) > self.p.period:

            for i in self.combs_stats:
                allowed_new_pos = True
                self.a = i[0][0]
                self.b = i[0][1]

                self.sp =  str(self.a+'/'+self.b)

                self.data0 = self.data_dic[self.a]
                self.data1 = self.data_dic[self.b]

                self.transform = i[1]
                self.zscore = i[2]
                self.beta= i[3]
                self.coint = i[4]



                # print(self.data0[0] ,self.beta[0],self.data1[0])
                self.spread = self.data0[0] -(self.beta[0]*self.data1[0])

                # self.spread_mean = btind.MovingAverageSimple(self.spread, period=self.p.period)





                self.plts[self.sp].append({'date':self.data0.datetime.datetime().strftime('%Y%m%d'),
                                           'spread':self.spread,
                                           'beta':self.beta[0],
                                           'coint_score':self.coint.score[0],
                                           'coint_crit': self.coint.crit[0],
                                           'zscore':self.zscore[0],
                                           })


                df = pd.DataFrame(self.plts[self.sp])

                self.plts[self.sp][-1]['spread_mean'] = df['spread'].mean()
                self.plts[self.sp][-1]['std'] = df['spread'].std()


                # Checking signals built with StatsModel.API :
                # self.ols_transfo = btind.OLS_Transformation(self.data0.close[0], self.data1.close[0],period=self.p.period,plot=True)




                if abs(self.coint.score[0]) < abs(self.coint.crit[0]) :
                    # print('Cointegration broke')
                    if self.pos_traker[self.sp] != 0:
                        self.log('CLOSE POS %s, price = %.2f' % (self.a, self.data0.close[0]))
                        self.close(self.data0)
                        self.log('CLOSE POS %s, price = %.2f' % (self.b, self.data1.close[0]))
                        self.close(self.data1)
                        # self.status = 0
                        self.pos_traker[self.sp] = 0
                    return

                if self.orderid:
                    return  # if an order is active, no new orders are allowed

                open_pos = {x:y for x,y in self.pos_traker.items() if y!=0}

                for i in list(open_pos.keys()):
                    x, y = i.split('/')
                    if self.a in [x,y] or self.b in [x,y]:
                        allowed_new_pos = False

                if self.p.printout and self.pos_traker[self.sp] != 0:
                #   print('Self  len:', len(self))
                #   print(self.a,' len:', len(self.data0))
                #   print(self.b,' len:', len(self.data1))
                #   print('self.a len == self.b len:',
                #           len(self.a) == len(self.b))
                #
                    print('Date:', self.data0.datetime.datetime())
                    print(open_pos)
                #   print('Data1 dt:', self.data1.datetime.datetime())
                #
                    print('Beta is', self.beta[0])
                    print(self.beta[0],self.data0[0],self.data1[0])
                #   print('status is', self.status)
                #   print('zscore is', self.zscore[0])
                    print('coint is', self.coint.score[0],' ',self.coint.pvalue[0],' ', self.coint.crit[0])
                    print('Coint > [5%]:',
                      abs(self.coint.score[0]) > abs(self.coint.crit[0]))

                    # print(self.sp,{'Spread: ':self.transform.spread[0],'-2STD':self.transform.spread_negtwostd[0],'-1STD':self.transform.spread_negOnestd[0],'Mean':self.transform.spread_mean[0],'1STD':self.transform.spread_std[0],'2STD':self.transform.spread_twostd[0]})
                    print(self.sp, {'Zscore: ': self.zscore[0], 'Down': self.lower_limit,
                                'Down_mid': self.low_medium,
                                'Up_mid': self.up_medium, 'UP': self.upper_limit})



                hr = self.beta[0]

                # Step 2: Check conditions for SHORT & place the order
                # Checking the condition for SHORT
                # if (self.zscore[0] > self.upper_limit) and (self.status != 1):
                if (self.zscore[0] > self.upper_limit) and self.pos_traker[self.sp] != 1 and allowed_new_pos == True: #and self.transform.spread < self.transform.spread_twostd:#(self.status != 1):
                # if (self.transform.spread[0] > self.transform.spread_std[0]) and self.pos_traker[self.sp] != 1:#(self.status != 1):

                    # Calculating the number of shares for each stock
                    # value = 0.5 * self.portfolio_value  # Divide the cash equally

                    # if len(list(open_pos.keys())) > 0:
                    #     size = round((self.portfolio_value / len(list(open_pos.keys()))) / (abs(self.transform.spread_twostd)-abs(self.transform.spread[0])))
                    # else:
                    #     size = round(self.portfolio_value / (
                    #             abs(self.transform.spread_twostd) - abs(self.transform.spread[0])))


                    max_loss = 0.01 * self.portfolio_value
                    size = int(max_loss / abs(self.plts[self.sp][-1]['std']))


                    print('size: ',size)
                    x = size  # Find the number of shares for Stock1
                    y = round(size * hr)  # Find the number of shares for Stock2


                    # Placing the order
                    self.log('SELL CREATE %s, price = %.2f, qty = %d' % (self.a, self.data0.close[0], x))
                    self.sell(data=self.data0, size=round(x + self.qty1))  # Place an order for buying y + qty2 shares
                    self.log('BUY CREATE %s, price = %.2f, qty = %d' % (self.b, self.data1.close[0], y))
                    self.buy(data=self.data1, size=round(y + self.qty2))  # Place an order for selling x + qty1 shares

                    # Updating the counters with new value
                    self.qty1 = x  # The new open position quantity for Stock1 is x shares
                    self.qty2 = y  # The new open position quantity for Stock2 is y shares

                    # self.status = 1  # The current status is "short the spread"
                    self.pos_traker[self.sp] = 1

                    print(self.a,' QTY is', -x)
                    print(self.b,' QTY is', y)

                    # Step 3: Check conditions for LONG & place the order
                    # Checking the condition for LONG
                # elif (self.zscore[0] < self.lower_limit) and (self.status != 2):
                elif (self.zscore[0] < self.lower_limit) and self.pos_traker[self.sp] != 2 and allowed_new_pos == True: # and self.transform.spread > self.transform.spread_negtwostd:
                # elif (self.transform.spread[0] < self.transform.spread_twostd[0]) and self.pos_traker[self.sp] != 2:

                    # Calculating the number of shares for each stock
                    # value = 0.5 * self.portfolio_value  # Divide the cash equally
                    max_loss = 0.01 * self.portfolio_value
                    size = int(max_loss / abs(self.plts[self.sp][-1]['std']))

                    print('size: ', size)
                    x = size  # Find the number of shares for Stock1
                    y = int(size * hr)  # Find the number of shares for Stock2


                    # Place the order
                    self.log('BUY CREATE %s, price = %.2f, qty = %d' % (self.a, self.data0.close[0], x))
                    self.buy(data=self.data0, size=round(x + self.qty1))  # Place an order for buying x + qty1 shares
                    self.log('SELL CREATE %s, price = %.2f, qty = %d' % (self.b, self.data1.close[0], y))
                    self.sell(data=self.data1, size=round(y + self.qty2))  # Place an order for selling y + qty2 shares

                    # Updating the counters with new value
                    self.qty1 = x  # The new open position quantity for Stock1 is x shares
                    self.qty2 = y  # The new open position quantity for Stock2 is y shares
                    # self.status = 2  # The current status is "long the spread"
                    self.pos_traker[self.sp] = 2

                    print(self.a, ' QTY is', x)
                    print(self.b, ' QTY is', -y)


                    # Step 4: Check conditions for No Trade
                    # If the z-score is within the two bounds, close all
                # elif (self.zscore[0] < self.up_medium and self.zscore[0] > self.low_medium):
                # elif (self.transform.spread > 0.1 *self.transform.spread_negOnestd and self.transform.spread < 0.1*self.transform.spread_std):


                elif self.pos_traker[self.sp]==1 and self.zscore[0] < self.up_medium or self.pos_traker[self.sp]==2 and self.zscore[0] > self.low_medium:

                    self.log('CLOSE POS %s, price = %.2f' % (self.a, self.data0.close[0]))
                    self.close(self.data0)
                    self.log('CLOSE POS %s, price = %.2f' % (self.b, self.data1.close[0]))
                    self.close(self.data1)
                    # self.status = 0
                    self.pos_traker[self.sp] = 0

                    self.qty1 = 0  # The new open position quantity for Stock1 is x shares
                    self.qty2 = 0  # The new open position quantity for Stock2 is y shares

                    print(self.a, ' QTY is', 0)
                    print(self.b, ' QTY is', 0)


                #stop loss
                # elif (self.transform.spread < self.transform.spread_negtwostd and self.pos_traker[self.sp] != 0) or( self.transform.spread > self.transform.spread_twostd  and self.pos_traker[self.sp] != 0):
                elif self.zscore[0] < self.stop_down or self.zscore[0] > self.stop_up:

                    self.log('Stop Loss %s, price = %.2f' % (self.a, self.data0.close[0]))
                    self.close(self.data0)
                    self.log('Stop Loss %s, price = %.2f' % (self.b, self.data1.close[0]))
                    self.close(self.data1)
                    # self.status = 0
                    self.pos_traker[self.sp] = 0

                    self.qty1 = 0  # The new open position quantity for Stock1 is x shares
                    self.qty2 = 0  # The new open position quantity for Stock2 is y shares

                    print(self.a, ' QTY is', 0)
                    print(self.b, ' QTY is', 0)





    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')




        for i in self.plts.keys():

            plot_df = pd.DataFrame(self.plts[i])

            years = mdates.YearLocator()
            years_fmt = mdates.DateFormatter('%Y')


            fig, axs = plt.subplots(4, sharex=True)
            fig.autofmt_xdate()
            fig.suptitle(i)
            axs[0].plot(plot_df['date'],plot_df['spread'])
            axs[0].plot(plot_df['date'], plot_df['spread_mean'], color='r')
            axs[0].plot(plot_df['date'], plot_df['spread_mean']+plot_df['std'], color='g')
            axs[0].plot(plot_df['date'], plot_df['spread_mean'] - plot_df['std'], color='g')
            axs[0].plot(plot_df['date'], plot_df['spread_mean']+(2*plot_df['std']), color='y')
            axs[0].plot(plot_df['date'], plot_df['spread_mean'] - (2*plot_df['std']), color='y')
            axs[0].set_title('Spread')
            axs[1].plot(plot_df['date'],plot_df['beta'])
            axs[1].set_title('Beta')
            axs[2].plot(plot_df['date'],abs(plot_df['coint_score']), color='b')
            axs[2].plot(plot_df['date'], abs(plot_df['coint_crit']), color='r')
            axs[2].set_title('Coint')
            axs[3].plot(plot_df['date'], plot_df['zscore'])
            axs[3].axhline(y = self.up_medium, color='g')
            axs[3].axhline(y=self.low_medium, color='g')
            axs[3].axhline(y = self.upper_limit, color='b')
            axs[3].axhline(y=self.lower_limit, color='b')
            axs[3].axhline(y = self.stop_up, color='r')
            axs[3].axhline(y=self.stop_down, color='r')



            axs[3].set_title('Zscore')
            axs[3].xaxis.set_major_locator(years)
            axs[3].xaxis.set_major_formatter(years_fmt)


            # ax1 = plt.subplot(311)
            # plt.title(i)
            # plt.plot(plot_df['date'],plot_df['spread'])
            # plt.setp(ax1.get_xticklabels(), visible=False)
            #
            # # share x only
            # ax2 = plt.subplot(312, sharex=ax1)
            # plt.title('Beta')
            # plt.plot(plot_df['date'],plot_df['beta'])
            # plt.setp(ax2.get_xticklabels(), visible=False)
            #
            # ax3 = plt.subplot(313, sharex=ax1)
            # plt.title('Coint')
            # plt.plot(plot_df['date'],plot_df['coint'])
            # plt.setp(ax3.get_xticklabels(), visible=False)
            #
            # ax4 = plt.subplot(411, sharex=ax1)
            # plt.title('Zscore')
            # plt.plot(plot_df['date'], plot_df['zscore'])


            plt.tight_layout()


            plt.show()





def runstrategy(ticker_list,bench_ticker):
    args = parse_args()
    print(args)

    # Create a cerebro
    cerebro = bt.Cerebro()

    # Get the dates from the args
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

    bench = bt.feeds.YahooFinanceData(
        dataname=bench_ticker,
        fromdate=fromdate,
        todate=todate,
        buffered=True,plot = False
    )

    cerebro.adddata(bench, name=bench_ticker)

    for i in ticker_list:
        print('Loading data: '+ i)
        data = bt.feeds.YahooFinanceData(
            dataname=i,
            fromdate=fromdate,
            todate=todate,
            buffered=True, plot = False
            )


        cerebro.adddata(data,name = i)




    # Add the strategy
    cerebro.addstrategy(PairTradingStrategy,
                        period=args.period,
                        stake=args.stake)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcash(args.cash)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcommission(commission=args.commperc)

    # And run it
    cerebro.run(runonce=not args.runnext,
                preload=not args.nopreload,
                oldsync=args.oldsync
                )

    # Plot if requested
    if args.plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red',figsize=(100,100))


def parse_args():
    parser = argparse.ArgumentParser(description='MultiData Strategy')

    parser.add_argument('--data0', '-d0',
                        default='../../datas/daily-PEP.csv',
                        help='1st data into the system')

    parser.add_argument('--data1', '-d1',
                        default='../../datas/daily-KO.csv',
                        help='2nd data into the system')

    parser.add_argument('--fromdate', '-f',
                        default='2018-01-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default='2020-09-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--period', default=60, type=int,
                        help='Period to apply to the Simple Moving Average')

    parser.add_argument('--cash', default=100000, type=int,
                        help='Starting Cash')

    parser.add_argument('--runnext', action='store_true',
                        help='Use next by next instead of runonce')

    parser.add_argument('--nopreload', action='store_true',
                        help='Do not preload the data')

    parser.add_argument('--oldsync', action='store_true',
                        help='Use old data synchronization method')

    parser.add_argument('--commperc', default=0.005, type=float,
                        help='Percentage commission (0.005 is 0.5%%')

    parser.add_argument('--stake', default=10, type=int,
                        help='Stake to apply in each operation')

    parser.add_argument('--plot', '-p', default=True, action='store_true',
                        help='Plot the read data')

    parser.add_argument('--numfigs', '-n', default=1,
                        help='Plot using numfigs figures')

    return parser.parse_args()

bench_ticker = 'SPY'
ticker_list = ['JPM','XLF','GS']




if __name__ == '__main__':
    runstrategy(ticker_list,bench_ticker)