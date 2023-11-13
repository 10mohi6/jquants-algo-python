import pandas as pd
import pytest

from jquants_algo import Algo


@pytest.fixture(scope="module", autouse=True)
def scope_module():
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

    yield MyAlgo(
        mail_address="dummy@dummy",
        password="dummy",
        outputs_dir_path="tests",
        data_dir_path="tests",
        ticker="6273",  # SMC
        size=100,  # 100 shares
    )


@pytest.fixture(scope="function", autouse=True)
def algo(scope_module, mocker):
    mocker.patch(
        "jquants_algo.Algo._get_prices_daily_quotes",
        return_value=pd.read_csv(
            "tests/6273-2023-11-13.csv", index_col=0, parse_dates=True
        ),
    )
    yield scope_module


# @pytest.mark.skip
def test_backtest(algo):
    algo.backtest()


# @pytest.mark.skip
def test_predict(algo):
    algo.predict()
