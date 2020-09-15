# coding: utf-8
# ##################################################################
# Pair Trading adapted to backtrader
# with PD.OLS and info for StatsModel.API
# author: Remi Roche
##################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import empyrical
import math


import matplotlib


from itertools import combinations
import matplotlib.dates as dates
from math import log
import matplotlib.pyplot as plt

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

pd.set_option('display.max_columns', 500)


def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    #Get the results we are interested in
    print(analyzer)
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total,2)
    strike_rate = (total_won / total_closed) * 100
    #Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate','Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed,total_won,total_lost]
    r2 = [str(round(strike_rate))+'%', win_streak, lose_streak, pnl_net]
    #Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    #Print the rows
    print_list = [h1,r1,h2,r2]
    row_format ="{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('',*row))

def printSQN(analyzer):
    sqn = round(analyzer.sqn,2)
    return 'SQN: {}'.format(sqn)


class FixedCommisionScheme(bt.CommInfoBase):
    '''
    This is a simple fixed commission scheme
    '''
    params = (
        ('commission', 1),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        )

    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission

#Traning Data
# High Vol:   2006-01-01  2013-01-01
# Low Vol : 2012-01-01  2014-08-01

high_vol = ['2006-01-01','2013-01-01']
low_vol = ['2012-01-01','2014-08-01']
total = ['2006-01-01','2020-08-01']



# Testing Data
# High Vol:   2017-06-01  2020-08-01
# Low Vol:    2015-12-01  2017-12-01


def parse_args():
    parser = argparse.ArgumentParser(description='MultiData Strategy')

    parser.add_argument('--data0', '-d0',
                        default='../../datas/daily-PEP.csv',
                        help='1st data into the system')

    parser.add_argument('--data1', '-d1',
                        default='../../datas/daily-KO.csv',
                        help='2nd data into the system')

    parser.add_argument('--fromdate', '-f',
                        default=high_vol[0],
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default=high_vol[1],
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--period', default=60, type=int,
                        help='Period to apply to the Simple Moving Average')

    parser.add_argument('--cash', default=10000, type=int,
                        help='Starting Cash')

    parser.add_argument('--runnext', action='store_true',
                        help='Use next by next instead of runonce')

    parser.add_argument('--nopreload', action='store_true',
                        help='Do not preload the data')

    parser.add_argument('--oldsync', action='store_true',
                        help='Use old data synchronization method')

    parser.add_argument('--commperc', default=0, type=float,
                        help='Percentage commission (0.005 is 0.5%%')

    parser.add_argument('--stake', default=0.01, type=int,
                        help='Stake to apply in each operation')

    parser.add_argument('--plot', '-p', default=True, action='store_true',
                        help='Plot the read data')

    parser.add_argument('--numfigs', '-n', default=1,
                        help='Plot using numfigs figures')

    parser.add_argument('--rf_rate', '-rf', default=0.001,
                        help='Risk free rate')

    return parser.parse_args()


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
    params = (('period', 60),)

    def next(self):
        y, x = (pd.Series(d.get(size=self.p.period)) for d in self.datas)
        #r_beta = sm.ols(y=y, x=x, window_type='full_sample')

        model_Simple = sm.OLS(y, x).fit()
        r_beta = model_Simple.params
        #print(r_beta)
        self.lines.beta[0] = r_beta
        self.lines.spread[0] = self.data0[0] - (self.lines.beta[0]*self.data1[0])

class PairTradingStrategy(bt.Strategy):
    args = parse_args()
    params = dict(
        period=args.period,
        stake=args.stake,
        qty1=0,
        qty2=0,
        printout=True,
        upper=1.7,
        lower=-1.7,
        up_medium=0.2,
        low_medium=-0.2,
        stop_up = 4,
        stop_down = -4,
        status=0,
        portfolio_value=args.cash,
        coint_pvalue_cutoff = 0.5,
        plot = args.plot
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

        print(list(enumerate(self.datas)))

        for i in list(enumerate(self.datas))[1:]:

            self.data_dic[i[1]._name] = i[1]






        self.combs = []
        # self.combs = list(self.data_dic.keys())

        for i in list(self.data_dic.keys())[1:]:
            self.combs.append([i,list(self.data_dic.keys())[0]])



        print(len(self.combs))
        print(self.combs)

        self.bench = list(enumerate(self.datas))[0][1]
        print(self.bench._name)

        self.combs_stats = []
        self.spread_traker = {}
        self.pos_traker = {}
        self.bench_beta = {}
        for a, b in self.combs:
            self.data0 = self.data_dic[a]
            self.data1 = self.data_dic[b]



            # Signals performed with PD.OLS :
            self.transform = btind.OLS_TransformationN(self.data0, self.data1,
                                                       period=self.p.period,
                                                       plot = False,
                                                       plotname = str(a+'-'+b+" Spread"),
                                                       subplot = True,
                                                       plotabove= True,
                                                       # plotmaster = self.bench
                                                       )
            self.zscore = self.transform.zscore

            self.beta =  btind.OLS_BetaN(self.data0, self.data1,
                                                       period=self.p.period,plot = False)

            self.bta = btind.OLS_BetaN(self.data0, self.bench,
                                       period=self.p.period, plot=False)

            self.bench_beta[a] = self.bta

            self.bta = btind.OLS_BetaN(self.data1, self.bench,
                                       period=self.p.period, plot=False)

            self.bench_beta[b] = self.bta

            # self.spread = spread_class(self.data0, self.data1, period=self.p.period)

            self.coint = btind.CointN(self.data0, self.data1,
                                        period=self.p.period,plot = False)

            self.combs_stats.append([[a,b],self.transform,self.zscore,self.beta,self.coint])

            self.pos_traker[a] = 0
            self.pos_traker[b] = 0

            self.spread_traker[str(a+'/'+b)] = [0,0,0,0]
            self.plts[str(a+'/'+b)] = [[],{'long':[[],[]],'short':[[],[]],'close':[[],[]],'stop':[[],[]]}]




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




                if self.zscore[0] < 10 and self.zscore[0] > -10:
                    self.plts[self.sp][0].append({'date':self.data0.datetime.datetime(),
                                               'spread':self.spread,
                                               'beta':self.beta[0],
                                               'coint_score':self.coint.score[0],
                                               'coint_crit': self.coint.crit[0],
                                                'coint_pvalue':self.coint.pvalue[0],
                                               'zscore':self.zscore[0],
                                               })
                else:
                    return


                df = pd.DataFrame(self.plts[self.sp][0])

                self.plts[self.sp][0][-1]['spread_mean'] = df['spread'].mean()
                self.plts[self.sp][0][-1]['std'] = df['spread'].std()



                if math.isnan(self.plts[self.sp][0][-1]['std']):
                    return

                # Checking signals built with StatsModel.API :
                # self.ols_transfo = btind.OLS_Transformation(self.data0.close[0], self.data1.close[0],period=self.p.period,plot=True)




                # if abs(self.coint.score[0]) < abs(self.coint.crit[0]) :
                if self.coint.pvalue[0] > self.p.coint_pvalue_cutoff:
                    # print('Cointegration broke')
                    if self.spread_traker[self.sp][0] != 0:


                        self.pos_traker[self.a] = self.pos_traker[self.a] - self.spread_traker[self.sp][2]
                        self.pos_traker[self.b] = self.pos_traker[self.b] - self.spread_traker[self.sp][3]



                        self.log('CLOSE POS %s, price = %.2f' % (self.a, self.data0.close[0]))
                        # self.close(self.data0)
                        self.log('CLOSE POS %s, price = %.2f' % (self.b, self.data1.close[0]))
                        # self.close(self.data1)
                        # self.status = 0
                        self.spread_traker[self.sp] = [0,0,0,0]
                        self.plts[self.sp][1]['close'][0].append(self.data0.datetime.datetime())
                        self.plts[self.sp][1]['close'][1].append(self.zscore[0])

                else:

                    if self.orderid:
                        return  # if an order is active, no new orders are allowed


                    if self.p.printout and self.spread_traker[self.sp][0] != 0:
                    #   print('Self  len:', len(self))
                    #   print(self.a,' len:', len(self.data0))
                    #   print(self.b,' len:', len(self.data1))
                    #   print('self.a len == self.b len:',
                    #           len(self.a) == len(self.b))
                    #
                    #     print('Date:', self.data0.datetime.datetime())
                    #
                    # #   print('Data1 dt:', self.data1.datetime.datetime())
                    # #
                    #     print('Beta is', self.beta[0])
                    #     print(self.beta[0],self.data0[0],self.data1[0])
                    # #   print('status is', self.status)
                    # #   print('zscore is', self.zscore[0])
                    #     print('coint is', self.coint.score[0],' ',self.coint.pvalue[0],' ', self.coint.crit[0])
                    #     print('Coint > [5%]:',
                    #       abs(self.coint.score[0]) > abs(self.coint.crit[0]))
                    #
                    #     # print(self.sp,{'Spread: ':self.transform.spread[0],'-2STD':self.transform.spread_negtwostd[0],'-1STD':self.transform.spread_negOnestd[0],'Mean':self.transform.spread_mean[0],'1STD':self.transform.spread_std[0],'2STD':self.transform.spread_twostd[0]})
                    #     print(self.sp, {'Zscore: ': self.zscore[0], 'Down': self.lower_limit,
                    #                 'Down_mid': self.low_medium,
                    #                 'Up_mid': self.up_medium, 'UP': self.upper_limit})
                        pass



                    hr = self.beta[0]

                    # Step 2: Check conditions for SHORT & place the order
                    # Checking the condition for SHORT
                    # if (self.zscore[0] > self.upper_limit) and (self.status != 1):
                    if (self.zscore[0] > self.upper_limit) and self.zscore[0] < self.stop_up and self.spread_traker[self.sp][0] == 0: #and self.transform.spread < self.transform.spread_twostd:#(self.status != 1):
                    # if (self.transform.spread[0] > self.transform.spread_std[0]) and self.pos_traker[self.sp] != 1:#(self.status != 1):

                        # Calculating the number of shares for each stock
                        # value = 0.5 * self.portfolio_value  # Divide the cash equally

                        # if len(list(open_pos.keys())) > 0:
                        #     size = round((self.portfolio_value / len(list(open_pos.keys()))) / (abs(self.transform.spread_twostd)-abs(self.transform.spread[0])))
                        # else:
                        #     size = round(self.portfolio_value / (
                        #             abs(self.transform.spread_twostd) - abs(self.transform.spread[0])))


                        max_loss = self.p.stake * self.portfolio_value
                        size = int(max_loss / abs(self.plts[self.sp][0][-1]['std']))


                        print('size: ',size)
                        x = size  # Find the number of shares for Stock1
                        y = round(size * hr)  # Find the number of shares for Stock2


                        # Placing the order
                        self.log('SELL CREATE %s, price = %.2f, qty = %d' % (self.a, self.data0.close[0], x))
                        # self.sell(data=self.data0, size=round(x + self.qty1))  # Place an order for buying y + qty2 shares
                        self.log('BUY CREATE %s, price = %.2f, qty = %d' % (self.b, self.data1.close[0], y))
                        # self.buy(data=self.data1, size=round(y + self.qty2))  # Place an order for selling x + qty1 shares

                        # Updating the counters with new value
                        self.qty1 = x  # The new open position quantity for Stock1 is x shares
                        self.qty2 = y  # The new open position quantity for Stock2 is y shares

                        # self.status = 1  # The current status is "short the spread"

                        self.pos_traker[self.a] = self.pos_traker[self.a] - x
                        self.pos_traker[self.b] = self.pos_traker[self.b] + y

                        self.spread_traker[self.sp] = [1,size,-x,y]

                        print(self.a,' QTY is', -x)
                        print(self.b,' QTY is', y)



                        self.plts[self.sp][1]['short'][0].append(self.data0.datetime.datetime())
                        self.plts[self.sp][1]['short'][1].append(self.zscore[0])

                        # Step 3: Check conditions for LONG & place the order
                        # Checking the condition for LONG
                    # elif (self.zscore[0] < self.lower_limit) and (self.status != 2):
                    elif (self.zscore[0] < self.lower_limit) and self.zscore[0] > self.stop_down and self.spread_traker[self.sp][0] == 0: # and self.transform.spread > self.transform.spread_negtwostd:
                    # elif (self.transform.spread[0] < self.transform.spread_twostd[0]) and self.pos_traker[self.sp] != 2:

                        # Calculating the number of shares for each stock
                        # value = 0.5 * self.portfolio_value  # Divide the cash equally
                        max_loss = self.p.stake * self.portfolio_value
                        size = int(max_loss / abs(self.plts[self.sp][0][-1]['std']))

                        print('size: ', size)
                        x = size  # Find the number of shares for Stock1
                        y = int(size * hr)  # Find the number of shares for Stock2


                        # Place the order
                        self.log('BUY CREATE %s, price = %.2f, qty = %d' % (self.a, self.data0.close[0], x))
                        # self.buy(data=self.data0, size=round(x + self.qty1))  # Place an order for buying x + qty1 shares
                        self.log('SELL CREATE %s, price = %.2f, qty = %d' % (self.b, self.data1.close[0], y))
                        # self.sell(data=self.data1, size=round(y + self.qty2))  # Place an order for selling y + qty2 shares

                        # Updating the counters with new value
                        self.qty1 = x  # The new open position quantity for Stock1 is x shares
                        self.qty2 = y  # The new open position quantity for Stock2 is y shares
                        # self.status = 2  # The current status is "long the spread"

                        self.pos_traker[self.a] = self.pos_traker[self.a] + x
                        self.pos_traker[self.b] = self.pos_traker[self.b] - y

                        self.spread_traker[self.sp] = [2,size,x,-y]

                        print(self.a, ' QTY is', x)
                        print(self.b, ' QTY is', -y)





                        self.plts[self.sp][1]['long'][0].append(self.data0.datetime.datetime())
                        self.plts[self.sp][1]['long'][1].append(self.zscore[0])


                        # Step 4: Check conditions for No Trade
                        # If the z-score is within the two bounds, close all
                    # elif (self.zscore[0] < self.up_medium and self.zscore[0] > self.low_medium):
                    # elif (self.transform.spread > 0.1 *self.transform.spread_negOnestd and self.transform.spread < 0.1*self.transform.spread_std):


                    elif self.spread_traker[self.sp][0]==1 and self.zscore[0] < self.up_medium or self.spread_traker[self.sp][0]==2 and self.zscore[0] > self.low_medium:

                        self.log('CLOSE POS %s, price = %.2f' % (self.a, self.data0.close[0]))
                        # self.close(self.data0)
                        self.log('CLOSE POS %s, price = %.2f' % (self.b, self.data1.close[0]))
                        # self.close(self.data1)
                        # self.status = 0


                        self.qty1 = 0  # The new open position quantity for Stock1 is x shares
                        self.qty2 = 0  # The new open position quantity for Stock2 is y shares

                        print(self.a, ' QTY is', 0)
                        print(self.b, ' QTY is', 0)

                        self.pos_traker[self.a] = self.pos_traker[self.a] - self.spread_traker[self.sp][2]
                        self.pos_traker[self.b] = self.pos_traker[self.b] - self.spread_traker[self.sp][3]

                        self.spread_traker[self.sp] = [0, 0, 0, 0]

                        self.plts[self.sp][1]['close'][0].append(self.data0.datetime.datetime())
                        self.plts[self.sp][1]['close'][1].append(self.zscore[0])


                    #stop loss
                    # elif (self.transform.spread < self.transform.spread_negtwostd and self.pos_traker[self.sp] != 0) or( self.transform.spread > self.transform.spread_twostd  and self.pos_traker[self.sp] != 0):
                    elif self.spread_traker[self.sp][0]!=0 and (self.zscore[0] < self.stop_down or self.zscore[0] > self.stop_up):

                        self.log('Stop Loss %s, price = %.2f' % (self.a, self.data0.close[0]))
                        # self.close(self.data0)
                        self.log('Stop Loss %s, price = %.2f' % (self.b, self.data1.close[0]))
                        # self.close(self.data1)
                        # self.status = 0


                        self.qty1 = 0  # The new open position quantity for Stock1 is x shares
                        self.qty2 = 0  # The new open position quantity for Stock2 is y shares

                        print(self.a, ' QTY is', 0)
                        print(self.b, ' QTY is', 0)

                        self.pos_traker[self.a] = self.pos_traker[self.a] - self.spread_traker[self.sp][2]
                        self.pos_traker[self.b] = self.pos_traker[self.b] - self.spread_traker[self.sp][3]

                        self.spread_traker[self.sp] = [0, 0, 0, 0]

                        self.plts[self.sp][1]['stop'][0].append(self.data0.datetime.datetime())
                        self.plts[self.sp][1]['stop'][1].append(self.zscore[0])


            print(self.pos_traker)
            print(self.spread_traker)


            db = []

            for i in self.pos_traker.keys():



                delta_beta = self.pos_traker[i] * self.bench_beta[i]
                db.append(delta_beta)

                print(i,' Position: ',self.getposition(data=self.data_dic[i]).size )
                order = self.pos_traker[i] - self.getposition(data=self.data_dic[i]).size
                print('Order: ', order  )

                if order > 0:
                    self.buy(data=self.data_dic[i], size=order)
                else:
                    self.sell(data=self.data_dic[i], size=abs(order))
            # print(db)
            db_sum = sum(db) + self.getposition(data=self.bench).size
            print('Delta Beta: ',db_sum )
            if db_sum <= -1:
                self.buy(data=self.bench, size=int(db_sum))
            elif db_sum >= 1:
                self.sell(data=self.bench, size=abs(int(db_sum)))

            print('Hedged ',self.getposition(data=self.bench).size, ' Shares SPY')




    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

        if self.p.plot:
            for i in self.plts.keys():

                plot_df = pd.DataFrame(self.plts[i][0])
                # print(i,' ',self.plts[i][1])
                # print(plot_df.tail())



                plt_dates = dates.date2num(plot_df['date'])



                fig, axs = plt.subplots(4, sharex=True)
                fig.autofmt_xdate()

                axs[0].plot(plt_dates,plot_df['spread'])
                axs[0].plot(plt_dates, plot_df['spread_mean'], color='r')
                axs[0].plot(plt_dates, plot_df['spread_mean']+plot_df['std'], color='g')
                axs[0].plot(plt_dates, plot_df['spread_mean'] - plot_df['std'], color='g')
                axs[0].plot(plt_dates, plot_df['spread_mean']+(2*plot_df['std']), color='y')
                axs[0].plot(plt_dates, plot_df['spread_mean'] - (2*plot_df['std']), color='y')
                axs[0].set_title(i+' Spread')

                axs[1].plot(plt_dates,plot_df['beta'])
                axs[1].set_title('Beta')

                # axs[2].plot(plt_dates,abs(plot_df['coint_score']), color='b')
                # axs[2].plot(plt_dates, abs(plot_df['coint_crit']), color='r')
                # axs[2].set_title('Coint')

                axs[2].xaxis_date()
                axs[2].plot(plt_dates, plot_df['zscore'])
                axs[2].axhline(y = self.up_medium, color='g')
                axs[2].axhline(y=self.low_medium, color='g')
                axs[2].axhline(y = self.upper_limit, color='b')
                axs[2].axhline(y=self.lower_limit, color='b')
                axs[2].axhline(y = self.stop_up, color='r')
                axs[2].axhline(y=self.stop_down, color='r')

                axs[2].scatter(self.plts[i][1]['long'][0], self.plts[i][1]['long'][1],500, color='green',marker = '^')
                axs[2].scatter(self.plts[i][1]['short'][0], self.plts[i][1]['short'][1],500, color='orange',marker = 'v')
                axs[2].scatter(self.plts[i][1]['close'][0], self.plts[i][1]['close'][1],500, color='blue',marker = 'd')
                axs[2].scatter(self.plts[i][1]['stop'][0], self.plts[i][1]['stop'][1],500, color='red',marker = 's')

                axs[2].format_xdata = dates.DateFormatter('%Y-%m-%d')

                axs[2].set_title('Zscore')
                # axs[3].xaxis.set_major_locator(years)
                axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))

                axs[3].plot(plt_dates, plot_df['coint_pvalue'])
                axs[3].set_title('Coint PValue')
                axs[3].axhline(y=self.p.coint_pvalue_cutoff, color='r')




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

    # bench = bt.feeds.YahooFinanceData(
    #     dataname=bench_ticker,
    #     fromdate=fromdate,
    #     todate=todate,
    #     buffered=True,plot = False
    # )

    bench = bt.feeds.GenericCSVData(
        dataname='/Users/joan/PycharmProjects/CSV_DB/IB/' + bench_ticker + '.csv',
        fromdate=fromdate,
        todate=todate,
        nullvalue=0.0,
        dtformat=('%Y%m%d'),
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        reverse=False,
        plot=False)

    cerebro.adddata(bench, name=bench_ticker)

    for i in ticker_list:
        print('Loading data: '+ i)
        # data = bt.feeds.YahooFinanceData(
        #     dataname=i,
        #     fromdate=fromdate,
        #     todate=todate,
        #     adjclose=True,
        #     buffered=True, plot = False
        #     )

        data = bt.feeds.GenericCSVData(
            dataname='/Users/joan/PycharmProjects/CSV_DB/IB/'+i+'.csv',
            fromdate=fromdate,
            todate=todate,
            nullvalue=0.0,
            dtformat=('%Y%m%d'),
            datetime=1,
            open=2,
            high=3,
            low=4,
            close=5,
            volume=6,
            reverse=False,
            plot= False)


        cerebro.adddata(data,name = i)




    # Add the strategy
    cerebro.addstrategy(PairTradingStrategy,
                        period=args.period,
                        stake=args.stake)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcash(args.cash)

    # Add the commission - only stocks like a for each operation
    # cerebro.broker.setcommission(commission=args.commperc)

    comminfo = FixedCommisionScheme()
    cerebro.broker.addcommissioninfo(comminfo)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='myysharpe', riskfreerate=args.rf_rate)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='mypyf')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days,
                        data=bench, _name='benchreturns')

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Benchmark,plot = False)
    cerebro.addobserver(bt.observers.DrawDown)

    # And run it
    strat = cerebro.run(runonce=not args.runnext,
                preload=not args.nopreload,
                oldsync=args.oldsync
                )

    # Plot if requested
    if args.plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red',figsize=(100,100))





    bench_returns = strat[0].analyzers.benchreturns.get_analysis()
    bench_df = pd.DataFrame.from_dict(bench_returns, orient='index', columns=['return'])
    return_df = pd.DataFrame.from_dict(strat[0].analyzers.mypyf.get_analysis()['returns'], orient='index',
                                       columns=['return'])

    # print('Sharpe Ratio(bt):', firstStrat.analyzers.myysharpe.get_analysis()['sharperatio'])
    # print('Sharpe Ratio:', empyrical.sharpe_ratio(return_df, risk_free=args.rf_rate / 252, period='daily')[0])
    # print('Sharpe Ratio Benchmark:', empyrical.sharpe_ratio(bench_df, risk_free=args.rf_rate / 252, period='daily')[0])
    # print('')
    #
    # print('Sortino Ratio:', empyrical.sortino_ratio(return_df, period='daily')[0])
    # print('Sortino Ratio Benchmark:', empyrical.sortino_ratio(bench_df, period='daily')[0])
    # print('')
    # print('VaR:', empyrical.value_at_risk(return_df) * 100, '%')
    # print('VaR Benchmark:', empyrical.value_at_risk(bench_df) * 100, '%')
    #
    # print('')
    #
    # print('Capture:', round(empyrical.capture(return_df, bench_df, period='daily')[0] * 100), '%')
    # print('')
    #
    # print('Max drawdown: ', round(empyrical.max_drawdown(return_df)[0] * 100), '%')
    # print('Max drawdown Benchmark: ', round(empyrical.max_drawdown(bench_df)[0] * 100), '%')
    #
    # print('')
    alpha, beta = empyrical.alpha_beta(return_df, bench_df, risk_free=args.rf_rate)
    # print('Beta: ', beta)
    # print('')
    # print('Annual return:', round(empyrical.annual_return(return_df)[0] * 100), '%')
    # print('Annual Vol:', round(empyrical.annual_volatility(return_df)[0] * 100), '%')
    # print('')
    # print('Annual return Benchmark:', round(empyrical.annual_return(bench_df)[0] * 100), '%')
    # print('Annual Vol Benchmark:', round(empyrical.annual_volatility(bench_df)[0] * 100), '%')
    # print('')

    dic = {'SQN': printSQN(strat[0].analyzers.sqn.get_analysis()),
            'sharpe': empyrical.sharpe_ratio(return_df, risk_free=args.rf_rate / 252, period='daily')[0],
           'sharpe_bm': empyrical.sharpe_ratio(bench_df, risk_free=args.rf_rate / 252, period='daily')[0],
           'sortino': empyrical.sortino_ratio(bench_df, period='daily')[0],
           'sortino_bm': empyrical.sortino_ratio(bench_df, period='daily')[0],
           'VaR': empyrical.value_at_risk(return_df) * 100,
           'VaR_bm': empyrical.value_at_risk(bench_df) * 100,
           'capture': round(empyrical.capture(return_df, bench_df, period='daily')[0] * 100),
           'max_dd': round(empyrical.max_drawdown(return_df)[0] * 100),
           'max_dd_bm':round(empyrical.max_drawdown(bench_df)[0] * 100),
           'beta': beta,
           'return_annual':round(empyrical.annual_return(return_df)[0] * 100,2),
           'return_annual_bm':round(empyrical.annual_volatility(return_df)[0] * 100,2),
           'vol_annual':round(empyrical.annual_return(bench_df)[0] * 100,2),
           'vol_annual_bm':round(empyrical.annual_volatility(bench_df)[0] * 100,2)}

    df = pd.DataFrame(dic,index = [0])
    print(df)

    def calc_stats(df):
        df['perc_ret'] = (1 + df['return']).cumprod() - 1
        # print(df.tail())
        return df

    s = return_df.rolling(30).std()
    b = bench_df.rolling(30).std()

    # Get final portfolio Value
    portvalue = cerebro.broker.getvalue()

    # Print out the final result
    print('Final Portfolio Value: ${}'.format(round(portvalue)), 'PnL: ${}'.format(round(portvalue - args.cash)),
          'PnL: {}%'.format(((portvalue / args.cash) - 1) * 100))

    # Finally plot the end results

    if args.plot:


        fig, axs = plt.subplots(2, sharex=True)
        fig.autofmt_xdate()

        axs[1].plot(s)
        axs[1].plot(b)

        axs[1].set_title('Drawdown')
        axs[1].legend(['Fund', 'Benchmark'])

        axs[0].set_title('Returns')
        axs[0].plot(calc_stats(return_df)['perc_ret'])
        axs[0].plot(calc_stats(bench_df)['perc_ret'])
        axs[0].legend(['Fund', 'Benchmark'])
        plt.show()



bench_ticker = 'SPY'
# ticker_list = ['XLF','BLK','WFC','BAC','JPM','GS','SPGI','AXP','MS','BK','MMC']
ticker_list = ['XLF','JPM','GS','MS','BAC','AXP']
# ticker_list = ['XLF','MS']
# ticker_list = ['VTI','XLF','XLU','XLK','XLV','XLY','XLP','XLE']



if __name__ == '__main__':
    runstrategy(ticker_list,bench_ticker)