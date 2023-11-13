# jquants-algo

[![PyPI](https://img.shields.io/pypi/v/jquants-algo)](https://pypi.org/project/jquants-algo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/jquants-algo-python/graph/badge.svg?token=X8QKKFK6AL)](https://codecov.io/gh/10mohi6/jquants-algo-python)
[![Python package](https://github.com/10mohi6/jquants-algo-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/10mohi6/jquants-algo-python/actions/workflows/python-package.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jquants-algo)](https://pypi.org/project/jquants-algo/)
[![Downloads](https://pepy.tech/badge/jquants-algo)](https://pepy.tech/project/jquants-algo)

jquants-algo is a python library for algorithmic trading with japanese stock trade using J-Quants on Python 3.8 and above.

## Installation

    $ pip install jquants-algo

## Usage

### backtest

```python
from jquants_algo import Algo
import pprint

class MyAlgo(Algo):
    def strategy(self):
        fast_ma = self.sma(period=3)
        slow_ma = self.sma(period=5)
        # golden cross
        self.sell_exit = self.buy_entry = (fast_ma > slow_ma) & (
            fast_ma.shift() <= slow_ma.shift()
        )
        # dead cross
        self.buy_exit = self.sell_entry = (fast_ma < slow_ma) & (
            fast_ma.shift() >= slow_ma.shift()
        )

algo = MyAlgo(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
)
pprint.pprint(algo.backtest())
```

![performance.png](https://raw.githubusercontent.com/10mohi6/jquants-algo-python/main/tests/7203-performance.png)

```python
{'long': {'average return': '0.156',
          'maximum drawdown': '49350.000',
          'profit': '11450.000',
          'profit factor': '1.080',
          'riskreward ratio': '1.455',
          'sharpe ratio': '0.038',
          'trades': '54.000',
          'win rate': '0.426'},
 'short': {'average return': '-0.238',
           'maximum drawdown': '42050.000',
           'profit': '-31020.000',
           'profit factor': '0.754',
           'riskreward ratio': '1.319',
           'sharpe ratio': '-0.091',
           'trades': '55.000',
           'win rate': '0.364'},
 'total': {'average return': '-0.043',
           'maximum drawdown': '79950.000',
           'profit': '-19570.000',
           'profit factor': '0.927',
           'riskreward ratio': '1.423',
           'sharpe ratio': '-0.013',
           'trades': '109.000',
           'win rate': '0.394'}}
```

### predict

```python
from jquants_algo import Algo
import pprint

class MyAlgo(Algo):
    def strategy(self):
        fast_ma = self.sma(period=3)
        slow_ma = self.sma(period=5)
        # golden cross
        self.sell_exit = self.buy_entry = (fast_ma > slow_ma) & (
            fast_ma.shift() <= slow_ma.shift()
        )
        # dead cross
        self.buy_exit = self.sell_entry = (fast_ma < slow_ma) & (
            fast_ma.shift() >= slow_ma.shift()
        )

algo = MyAlgo(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
)
pprint.pprint(algo.predict())
```

```python
{'buy entry': True,
 'buy exit': False,
 'close': 2416.5,
 'date': '2023-08-22',
 'sell entry': False,
 'sell exit': True}
```

### advanced

```python
from jquants_algo import Algo
import pprint

class MyAlgo(Algo):
    def strategy(self):
        rsi = self.rsi(period=10)
        ema = self.ema(period=20)
        lower = ema - (ema * 0.001)
        upper = ema + (ema * 0.001)
        self.buy_entry = (rsi < 30) & (self.df.Close < lower)
        self.sell_entry = (rsi > 70) & (self.df.Close > upper)
        self.sell_exit = ema > self.df.Close
        self.buy_exit = ema < self.df.Close

algo = MyAlgo(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
    outputs_dir_path="outputs",
    data_dir_path="data",
)
pprint.pprint(algo.backtest())
pprint.pprint(algo.predict())
```

## Supported indicators

- Simple Moving Average 'sma'
- Exponential Moving Average 'ema'
- Moving Average Convergence Divergence 'macd'
- Relative Strenght Index 'rsi'
- Bollinger Bands 'bbands'
- Market Momentum 'mom'
- Stochastic Oscillator 'stoch'
- Average True Range 'atr'

## Getting started

For help getting started with J-Quants, view our online [documentation](https://jpx-jquants.com/).
