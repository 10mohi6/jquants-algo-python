import datetime
import os
import warnings
from enum import IntEnum
from typing import Dict, Tuple

import jquantsapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(
    action="ignore", message="invalid value encountered in scalar divide"
)
warnings.filterwarnings(
    action="ignore", message="divide by zero encountered in scalar divide"
)


class Col(IntEnum):
    CLOSE = 0
    BUY_ENTRY = 1
    BUY_EXIT = 2
    SELL_ENTRY = 3
    SELL_EXIT = 4
    BUY_PROFIT = 5
    SELL_PROFIT = 6
    TOTAL_PROFIT = 7
    BUY_RETURN = 8
    SELL_RETURN = 9
    TOTAL_RETURN = 10
    LONG = 11
    SHORT = 12
    TOTAL = 13


class Algo(object):
    def __init__(
        self,
        *,
        outputs_dir_path: str = ".",
        data_dir_path: str = ".",
        mail_address: str = "",
        password: str = "",
        ticker: str = "",
        size: int = 1,
    ) -> None:
        self.ticker = ticker
        self.outputs_dir_path = outputs_dir_path
        self.data_dir_path = data_dir_path
        self.size = size
        self.cli = jquantsapi.Client(mail_address=mail_address, password=password)
        os.makedirs(self.outputs_dir_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)

    def strategy(self) -> None:
        pass

    def _get_prices_daily_quotes(self) -> pd.DataFrame:
        csv = "{}/{}-{}.csv".format(
            self.data_dir_path,
            self.ticker,
            datetime.date.today(),
        )
        if os.path.isfile(csv):
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
        else:
            df = (
                self.cli.get_prices_daily_quotes(self.ticker)[
                    [
                        "Date",
                        "AdjustmentOpen",
                        "AdjustmentHigh",
                        "AdjustmentLow",
                        "AdjustmentClose",
                        "AdjustmentVolume",
                    ]
                ]
                .rename(
                    columns={
                        "AdjustmentOpen": "Open",
                        "AdjustmentHigh": "High",
                        "AdjustmentLow": "Low",
                        "AdjustmentClose": "Close",
                        "AdjustmentVolume": "Volume",
                    }
                )
                .set_index("Date")
            ).fillna(method="ffill")
            df.to_csv(csv)
        return df

    def backtest(self) -> Dict:
        self.df = self._get_prices_daily_quotes()
        self.strategy()
        df = self.df[["Close"]]
        df = df.assign(buy_entry=self.buy_entry)
        df = df.assign(buy_exit=self.buy_exit)
        df = df.assign(sell_entry=self.sell_entry)
        df = df.assign(sell_exit=self.sell_exit)
        df["buy_profit"] = df["sell_profit"] = df["total_profit"] = df[
            "buy_return"
        ] = df["sell_return"] = df["total_return"] = df["Long"] = df["Short"] = df[
            "Total"
        ] = np.nan
        long = short = 0.0
        buy = sell = 0.0
        for i in range(len(df)):
            if df.iat[i, Col.BUY_ENTRY] and buy == 0.0:
                buy = df.iat[i, Col.CLOSE] * self.size
            elif df.iat[i, Col.BUY_EXIT] and buy > 0.0:
                df.iat[i, Col.BUY_PROFIT] = (df.iat[i, Col.CLOSE] * self.size) - buy
                df.iat[i, Col.BUY_RETURN] = (
                    ((df.iat[i, Col.CLOSE] * self.size) - buy) / buy * 100
                ).round(1)
                df.iat[i, Col.TOTAL_PROFIT] = (
                    df.iat[i, Col.BUY_PROFIT]
                    if np.isnan(df.iat[i, Col.TOTAL_PROFIT])
                    else df.iat[i, Col.TOTAL_PROFIT] + df.iat[i, Col.BUY_PROFIT]
                )
                df.iat[i, Col.TOTAL_RETURN] = (
                    df.iat[i, Col.BUY_RETURN]
                    if np.isnan(df.iat[i, Col.TOTAL_RETURN])
                    else df.iat[i, Col.TOTAL_RETURN] + df.iat[i, Col.BUY_RETURN]
                )
                long += df.iat[i, Col.BUY_PROFIT]
                buy = 0.0
            if df.iat[i, Col.SELL_ENTRY] and sell == 0.0:
                sell = df.iat[i, Col.CLOSE] * self.size
            elif df.iat[i, Col.SELL_EXIT] and sell > 0.0:
                df.iat[i, Col.SELL_PROFIT] = sell - (df.iat[i, Col.CLOSE] * self.size)
                df.iat[i, Col.SELL_RETURN] = (
                    (sell - (df.iat[i, Col.CLOSE] * self.size))
                    / (df.iat[i, Col.CLOSE] * self.size)
                    * 100
                ).round(1)
                df.iat[i, Col.TOTAL_PROFIT] = (
                    df.iat[i, Col.SELL_PROFIT]
                    if np.isnan(df.iat[i, Col.TOTAL_PROFIT])
                    else df.iat[i, Col.TOTAL_PROFIT] + df.iat[i, Col.SELL_PROFIT]
                )
                df.iat[i, Col.TOTAL_RETURN] = (
                    df.iat[i, Col.SELL_RETURN]
                    if np.isnan(df.iat[i, Col.TOTAL_RETURN])
                    else df.iat[i, Col.TOTAL_RETURN] + df.iat[i, Col.SELL_RETURN]
                )
                short += df.iat[i, Col.SELL_PROFIT]
                sell = 0.0
            df.iat[i, Col.LONG] = long
            df.iat[i, Col.SHORT] = short
            df.iat[i, Col.TOTAL] = long + short
        fig = plt.figure(figsize=((6.4 * 2.5), (4.8 * 2.5)))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        ax1.plot(df["Long"])
        ax1.plot(df["Short"])
        ax1.plot(df["Total"])
        ax1.legend(["long profit", "short profit", "total profit"])
        ax1.grid()
        ax2.plot(df["Close"] - df["Close"][0])
        ax2.legend(["close"])
        ax2.grid()
        ax3.hist(
            [df["buy_return"], df["sell_return"]],
            50,
            rwidth=0.8,
            label=["long return", "short return"],
        )
        ax3.axvline(
            (df["buy_return"].sum() + df["sell_return"].sum())
            / (df["buy_return"].count() + df["sell_return"].count()),
            linestyle="dashed",
            color="green",
            label="total average return",
            linewidth=2,
        )
        ax3.legend()
        # ax3.grid()
        fig.suptitle(self.ticker)
        fig.tight_layout()
        plt.savefig("{}/performance.png".format(self.outputs_dir_path))
        plt.clf()
        plt.close()
        df.to_csv("{}/performance.csv".format(self.outputs_dir_path))
        rd = {"long": {}, "short": {}, "total": {}}
        # long
        long_trades = df["buy_profit"].count()
        long_win_num = df.query("buy_profit>0").count()["buy_profit"]
        long_loss_num = long_trades - long_win_num
        long_win = df.query("buy_profit>0").sum()["buy_profit"]
        long_loss = df.query("buy_profit<0").sum()["buy_profit"]
        long_average_win = abs(long_win) / long_win_num
        long_average_loss = abs(long_loss) / long_loss_num
        long_average_return = df["buy_return"].mean()
        long_sharpe_ratio = df["buy_return"].mean() / df["buy_return"].std()
        long_mdd = (np.maximum.accumulate(df["Long"]) - df["Long"]).max()
        rd["long"]["profit"] = "{:.3f}".format(df["Long"][-1])
        rd["long"]["trades"] = "{:.3f}".format(long_trades)
        rd["long"]["win rate"] = "{:.3f}".format(long_win_num / long_trades)
        rd["long"]["profit factor"] = "{:.3f}".format(abs(long_win) / abs(long_loss))
        rd["long"]["riskreward ratio"] = "{:.3f}".format(
            long_average_win / long_average_loss
        )
        rd["long"]["average return"] = "{:.3f}".format(long_average_return)
        rd["long"]["sharpe ratio"] = "{:.3f}".format(long_sharpe_ratio)
        rd["long"]["maximum drawdown"] = "{:.3f}".format(long_mdd)
        # short
        short_trades = df["sell_profit"].count()
        short_win_num = df.query("sell_profit>0").count()["sell_profit"]
        short_loss_num = short_trades - short_win_num
        short_win = df.query("sell_profit>0").sum()["sell_profit"]
        short_loss = df.query("sell_profit<0").sum()["sell_profit"]
        short_average_win = abs(short_win) / short_win_num
        short_average_loss = abs(short_loss) / short_loss_num
        short_average_return = df["sell_return"].mean()
        short_sharpe_ratio = df["sell_return"].mean() / df["sell_return"].std()
        short_mdd = (np.maximum.accumulate(df["Short"]) - df["Short"]).max()
        rd["short"]["profit"] = "{:.3f}".format(df["Short"][-1])
        rd["short"]["trades"] = "{:.3f}".format(short_trades)
        rd["short"]["win rate"] = "{:.3f}".format(short_win_num / short_trades)
        rd["short"]["profit factor"] = "{:.3f}".format(abs(short_win) / abs(short_loss))
        rd["short"]["riskreward ratio"] = "{:.3f}".format(
            short_average_win / short_average_loss
        )
        rd["short"]["average return"] = "{:.3f}".format(short_average_return)
        rd["short"]["sharpe ratio"] = "{:.3f}".format(short_sharpe_ratio)
        rd["short"]["maximum drawdown"] = "{:.3f}".format(short_mdd)
        # total
        total_trades = df["total_profit"].count()
        total_win_num = df.query("total_profit>0").count()["total_profit"]
        total_loss_num = total_trades - total_win_num
        total_win = df.query("total_profit>0").sum()["total_profit"]
        total_loss = df.query("total_profit<0").sum()["total_profit"]
        total_average_win = abs(total_win) / total_win_num
        total_average_loss = abs(total_loss) / total_loss_num
        total_average_return = df["total_return"].mean()
        total_sharpe_ratio = df["total_return"].mean() / df["total_return"].std()
        total_mdd = (np.maximum.accumulate(df["Total"]) - df["Total"]).max()
        rd["total"]["profit"] = "{:.3f}".format(df["Total"][-1])
        rd["total"]["trades"] = "{:.3f}".format(total_trades)
        rd["total"]["win rate"] = "{:.3f}".format(total_win_num / total_trades)
        rd["total"]["profit factor"] = "{:.3f}".format(abs(total_win) / abs(total_loss))
        rd["total"]["riskreward ratio"] = "{:.3f}".format(
            total_average_win / total_average_loss
        )
        rd["total"]["average return"] = "{:.3f}".format(total_average_return)
        rd["total"]["sharpe ratio"] = "{:.3f}".format(total_sharpe_ratio)
        rd["total"]["maximum drawdown"] = "{:.3f}".format(total_mdd)
        return rd

    def sma(self, *, period: int) -> pd.DataFrame:
        return self.df.Close.rolling(period).mean()

    def ema(self, *, period: int) -> pd.DataFrame:
        return self.df.Close.ewm(span=period).mean()

    def bbands(
        self, *, period: int = 20, band: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        std = self.df.Close.rolling(period).std()
        mean = self.df.Close.rolling(period).mean()
        return mean + (std * band), mean, mean - (std * band)

    def macd(
        self,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macd = (
            self.df.Close.ewm(span=fast_period).mean()
            - self.df.Close.ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    def stoch(
        self, *, k_period: int = 5, d_period: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k = (
            (self.df.Close - self.df.Low.rolling(k_period).min())
            / (
                self.df.High.rolling(k_period).max()
                - self.df.Low.rolling(k_period).min()
            )
            * 100
        )
        d = k.rolling(d_period).mean()
        return k, d

    def rsi(self, *, period: int = 14) -> pd.DataFrame:
        return 100 - 100 / (
            1
            - self.df.Close.diff().clip(lower=0).rolling(period).mean()
            / self.df.Close.diff().clip(upper=0).rolling(period).mean()
        )

    def atr(self, *, period: int = 14) -> pd.DataFrame:
        a = (self.df.High - self.df.Low).abs()
        b = (self.df.High - self.df.Close.shift()).abs()
        c = (self.df.Low - self.df.Close.shift()).abs()

        df = pd.concat([a, b, c], axis=1).max(axis=1)
        return df.ewm(span=period).mean()

    def mom(self, *, period: int = 10) -> pd.DataFrame:
        return self.df.Close.diff(period)

    def predict(self) -> Dict:
        self.df = self._get_prices_daily_quotes()
        self.strategy()
        df = self.df[["Close"]]
        df = df.assign(buy_entry=self.buy_entry)
        df = df.assign(buy_exit=self.buy_exit)
        df = df.assign(sell_entry=self.sell_entry)
        df = df.assign(sell_exit=self.sell_exit)
        return {
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "close": df.iloc[-1]["Close"],
            "buy entry": df.iloc[-1]["buy_entry"],
            "buy exit": df.iloc[-1]["buy_exit"],
            "sell entry": df.iloc[-1]["sell_entry"],
            "sell exit": df.iloc[-1]["sell_exit"],
        }
