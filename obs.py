#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2020 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import uuid

from backtrader import Observer



class Spread(Observer,spread):




    lines = ('Spread')

    params = dict(stds = [1])

    plotinfo = dict(plot=True, subplot=True,
                    plotname='Trades - Net Profit/Loss',
                    plotymargin=0.10,
                    plothlines=[0.0])

    plotlines = dict(
        pnlplus=dict(_name='Positive',
                     ls='', marker='o', color='blue',
                     markersize=8.0, fillstyle='full'),
        pnlminus=dict(_name='Negative',
                      ls='', marker='o', color='red',
                      markersize=8.0, fillstyle='full')
    )

    def __init__(self):




        pass

    def next(self):







        # for trade in self._owner._tradespending:
        #     if trade.data not in self.ddatas:
        #         continue
        #
        #     if not trade.isclosed:
        #         continue
        #
        #     pnl = trade.pnlcomm if self.p.pnlcomm else trade.pnl
        #
        #     if pnl >= 0.0:
        #         self.lines.pnlplus[0] = pnl
        #     else:
        #         self.lines.pnlminus[0] = pnl
