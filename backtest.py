import backtrader as bt
from datetime import datetime
from collections import OrderedDict
import pandas as pd
from math import sqrt
import empyrical

import matplotlib.pyplot as plt


futures_like = False

if futures_like:
    commission, margin, mult = 2.0, 2000.0, 10.0
else:
    commission, margin, mult = 0.005, None, 1


class firstStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.opsize = order.executed.size
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

                gross_pnl = (order.executed.price - self.buyprice) * \
                    self.opsize
                if margin:
                    gross_pnl *= mult
                net_pnl = gross_pnl - self.buycomm - order.executed.comm
                self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                         (gross_pnl, net_pnl))
    def __init__(self):
        # self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)

        self.sma_long = bt.indicators.SMA(self.data.close,period = 75)
        self.sma_short = bt.indicators.SMA(self.data.close,period = 30)
        self.bb = bt.indicators.BBands(self.data.close, period=100)
        # self.sh = bt.analyzers.SharpeRatio(self.data.close,riskfreerate = rf)




    # RSI Strat tst
    # def next(self):
    #     if not self.position:
    #         if self.rsi < 20:
    #             self.log('BUY CREATE, %.2f' % self.data.close[0])
    #             self.buy(size=50)
    #     else:
    #         if self.rsi > 80:
    #             self.log('SELL CREATE, %.2f' % self.data.close[0])
    #             self.sell(size=50)
    # BB strat test
    # def next(self):
    #     if not self.position:
    #         if self.data.close[0] >self.bb.lines.mid and self.data.close[0] < self.bb.lines.mid +1:
    #             self.log('BUY CREATE, %.2f' % self.data.close[0])
    #             self.order_target_percent(target=1)
    #     else:
    #         if self.data.close[0] > self.bb.lines.top or self.data.close[0] < self.bb.lines.mid:
    #             self.log('SELL CREATE, %.2f' % self.data.close[0])
    #             self.order_target_percent(target=0)

    # SMA cross strat
    def next(self):

        if not self.position:
            self.order_target_percent(data=bond, target=0.3)
            if self.sma_long <self.sma_short:
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.order_target_percent(data = data,target=0.7)



        else:
            if self.sma_long >self.sma_short:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.order_target_percent(data = data,target=0)






    # Monthly rebalance
    # def start(self):
    #
    #     # Add a timer which will be called on the 1st trading day of the month
    #     self.add_timer(
    #         bt.timer.SESSION_END,  # when it will be called
    #         monthdays=[1],  # called on the 1st day of the month
    #         monthcarry=True,  # called on the 2nd day if the 1st is holiday
    #     )
    #
    # def notify_timer(self, timer, when, *args, **kwargs):
    #     # Add the influx of monthly cash to the broker
    #

    #
    #     # buy available cash
    #
    #     self.order_target_value(data = data,target=self.broker.get_value()*0.33)
    #     self.order_target_value(data=bond, target=self.broker.get_value()*0.33)
    #     self.order_target_value(data=gold, target=self.broker.get_value()*0.33)
    #
    # def stop(self):
    #     # calculate the actual returns
    #     self.order_target_value(target=0)







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
    print('SQN: {}'.format(sqn))



rf = 0.018
#Variable for our starting cash
startcash = 10000







#Create an instance of cerebro
cerebro = bt.Cerebro()

#Add our strategy
cerebro.addstrategy(firstStrategy)



plots = True

asset = 'F'
bond_ass = 'IEF'
gold_ass = 'GLD'
bench_asset = 'SPY'



startdate = datetime(2015,7,1)

enddate = datetime(2019,7,1)
# Get data from Yahoo Finance
# store = bt.stores.IBStore(port=4001)
# data = store.getdata(
#     dataname=asset,
#     fromdate = startdate,
#     todate = enddate,
#     buffered= True
#      )

# Get data from Yahoo Finance.
data = bt.feeds.YahooFinanceData(
    dataname=asset,
    fromdate = startdate,
    todate = enddate,
    buffered= True
    )

bond = bt.feeds.YahooFinanceData(
    dataname= bond_ass,
    fromdate = startdate,
    todate = enddate,
    buffered= True
    )

# gold = bt.feeds.YahooFinanceData(
#     dataname= gold_ass,
#     fromdate = startdate,
#     todate = enddate,
#     buffered= True
#     )


benchmark = bt.feeds.YahooFinanceData(
    dataname=bench_asset,
    fromdate = startdate,
    todate = enddate,
    buffered= True
    )

#Add the data to Cerebro
cerebro.adddata(data)

cerebro.adddata(bond)

# cerebro.adddata(gold)

cerebro.adddata(benchmark)




# Set our desired cash start
cerebro.broker.setcash(startcash)

# Add the analyzers we are interested in
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='myysharpe',riskfreerate=rf)
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='mypyf')
cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days,
                    data=benchmark, _name='benchreturns')


cerebro.addobserver(bt.observers.Value)
cerebro.addobserver(bt.observers.Benchmark)
cerebro.addobserver(bt.observers.DrawDown)

cerebro.broker.setcommission(
    commission=commission, margin=margin, mult=mult)


# Run over everything
strategies = cerebro.run()
firstStrat = strategies[0]

# print the analyzers
# printTradeAnalysis(firstStrat.analyzers.ta.get_analysis())
printSQN(firstStrat.analyzers.sqn.get_analysis())

# print(firstStrat.analyzers.myysharpe.get_analysis())


bench_returns = firstStrat.analyzers.benchreturns.get_analysis()
bench_df = pd.DataFrame.from_dict(bench_returns, orient='index', columns = ['return'])
return_df = pd.DataFrame.from_dict(firstStrat.analyzers.mypyf.get_analysis()['returns'], orient='index', columns = ['return'])

# print('Sharpe Ratio(bt):', firstStrat.analyzers.myysharpe.get_analysis()['sharperatio'])
print('Sharpe Ratio:', empyrical.sharpe_ratio(return_df,risk_free= rf/252,period='daily')[0])
print('Sharpe Ratio Benchmark:', empyrical.sharpe_ratio(bench_df,risk_free= rf/252,period='daily')[0])
print('')

print('Sortino Ratio:', empyrical.sortino_ratio(return_df,period='daily')[0])
print('Sortino Ratio Benchmark:', empyrical.sortino_ratio(bench_df,period='daily')[0])
print('')
print('VaR:', empyrical.value_at_risk(return_df)*100,'%')
print('VaR Benchmark:', empyrical.value_at_risk(bench_df)*100,'%')

print('')

print('Capture:', round(empyrical.capture(return_df,bench_df,period='daily')[0]*100),'%')
print('')

print('Max drawdown: ', round(empyrical.max_drawdown(return_df)[0]*100),'%')
print('Max drawdown Benchmark: ', round(empyrical.max_drawdown(bench_df)[0]*100),'%')

print('')
alpha, beta = empyrical.alpha_beta(return_df,bench_df,risk_free=rf)
print('Beta: ', beta)
print('')
print('Annual return:', round(empyrical.annual_return(return_df)[0]*100),'%')
print('Annual Vol:', round(empyrical.annual_volatility(return_df)[0]*100),'%')
print('')
print('Annual return Benchmark:', round(empyrical.annual_return(bench_df)[0]*100),'%')
print('Annual Vol Benchmark:', round(empyrical.annual_volatility(bench_df)[0]*100),'%')
print('')


def calc_stats(df):
    df['perc_ret'] = (1 + df['return']).cumprod() - 1
    # print(df.tail())
    return df



s = return_df.rolling(30).std()
b = bench_df.rolling(30).std()


#Get final portfolio Value
portvalue = cerebro.broker.getvalue()

#Print out the final result
print('Final Portfolio Value: ${}'.format(round(portvalue)), 'PnL: ${}'.format(round(portvalue-startcash)),'PnL: {}%'.format(((portvalue/startcash)-1)*100) )

#Finally plot the end results


if plots == True:
    plt.plot(s)
    plt.plot(b)
    plt.legend(['Fund','Benchmark'])
    plt.show()

    plt.plot(calc_stats(return_df)['perc_ret'])
    plt.plot(calc_stats(bench_df)['perc_ret'])
    plt.legend(['Fund','Benchmark'])
    plt.show()

    cerebro.plot(style='candlestick', barup='green', bardown='red')