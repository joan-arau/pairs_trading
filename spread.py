from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import backtrader.indicators as btind


class Myspread(bt.Indicator):

    lines = ('trix',)
    params = (('period', 150),)

    def __init__(self):

        self.lines.spread = self.datas[0]-self.datas[1]
