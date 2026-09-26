from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


TRADING_DAYS_PER_YEAR = 252


@dataclass
class AssetAnalyzer:
    """Analyze a single asset's historical price series."""

    prices: pd.Series
    name: str = "Asset"

    def __post_init__(self):
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index().dropna()
        self.prices.name = self.name

    @classmethod
    def from_yfinance(cls, ticker: str, start: Optional[str] = None,
                      end: Optional[str] = None, name: Optional[str] = None) -> "AssetAnalyzer":
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for from_yfinance(). Install it with "
                "'pip install yfinance'."
            ) from e

        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. Check the symbol, date "
                f"range, and that this machine has internet access to Yahoo Finance."
            )
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return cls(prices=close, name=name or ticker)

    @classmethod
    def from_csv(cls, path: str, date_col: str = "Date", price_col: str = "Close",
                 name: Optional[str] = None) -> "AssetAnalyzer":
        df = pd.read_csv(path, parse_dates=[date_col])
        series = df.set_index(date_col)[price_col]
        return cls(prices=series, name=name or Path(path).stem)

    @property
    def returns(self) -> pd.Series:
        """Simple (arithmetic) daily returns."""
        return self.prices.pct_change().dropna()

    @property
    def log_returns(self) -> pd.Series:
        """Log daily returns: ln(P_t / P_{t-1})."""
        return np.log(self.prices / self.prices.shift(1)).dropna()

    def cumulative_returns(self) -> pd.Series:
        """Growth of $1 invested at the start of the series."""
        return (1 + self.returns).cumprod()

    def annualized_return(self, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Geometric (compounded) annualized return."""
        r = self.returns
        n = len(r)
        if n == 0:
            return float("nan")
        total_growth = (1 + r).prod()
        years = n / periods_per_year
        return total_growth ** (1 / years) - 1

    def annualized_volatility(self, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Annualized standard deviation of returns."""
        return self.returns.std(ddof=1) * np.sqrt(periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Annualized Sharpe ratio; risk_free_rate is an annual rate."""
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = self.returns - rf_per_period
        std = excess.std(ddof=1)
        if std == 0:
            return float("nan")
        return (excess.mean() / std) * np.sqrt(periods_per_year)

    def sortino_ratio(self, risk_free_rate: float = 0.0,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Like Sharpe, but only penalizes downside volatility."""
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = self.returns - rf_per_period
        downside = excess[excess < 0]
        downside_std = downside.std(ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return float("nan")
        return (excess.mean() / downside_std) * np.sqrt(periods_per_year)

    def max_drawdown(self) -> dict:
        """Return maximum peak-to-trough decline and the peak/trough dates."""
        cum = self.cumulative_returns()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        trough_date = drawdown.idxmin()
        peak_date = cum.loc[:trough_date].idxmax()
        return {
            "max_drawdown": drawdown.min(),
            "peak_date": peak_date,
            "trough_date": trough_date,
        }

    def drawdown_series(self) -> pd.Series:
        cum = self.cumulative_returns()
        running_max = cum.cummax()
        return cum / running_max - 1

    def value_at_risk(self, confidence: float = 0.95, method: str = "historical",
                      horizon_days: int = 1) -> float:
        """Return VaR as a positive fraction of portfolio value."""
        r = self.returns
        alpha = 1 - confidence
        if method == "historical":
            var_1d = -np.quantile(r, alpha)
        elif method == "parametric":
            mu, sigma = r.mean(), r.std(ddof=1)
            from scipy.stats import norm
            var_1d = -(mu + sigma * norm.ppf(alpha))
        else:
            raise ValueError("method must be 'historical' or 'parametric'")
        return var_1d * np.sqrt(horizon_days)

    def moving_average(self, window: int) -> pd.Series:
        return self.prices.rolling(window).mean()

    def exponential_moving_average(self, span: int) -> pd.Series:
        return self.prices.ewm(span=span, adjust=False).mean()

    def rsi(self, window: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing)."""
        delta = self.prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        mid = self.moving_average(window)
        std = self.prices.rolling(window).std()
        return pd.DataFrame({
            "mid": mid,
            "upper": mid + num_std * std,
            "lower": mid - num_std * std,
        })

    def summary(self, risk_free_rate: float = 0.0) -> dict:
        dd = self.max_drawdown()
        return {
            "name": self.name,
            "start_date": self.prices.index[0].date().isoformat(),
            "end_date": self.prices.index[-1].date().isoformat(),
            "num_observations": len(self.prices),
            "start_price": round(float(self.prices.iloc[0]), 4),
            "end_price": round(float(self.prices.iloc[-1]), 4),
            "total_return_pct": round((self.prices.iloc[-1] / self.prices.iloc[0] - 1) * 100, 2),
            "annualized_return_pct": round(self.annualized_return() * 100, 2),
            "annualized_volatility_pct": round(self.annualized_volatility() * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio(risk_free_rate), 3),
            "sortino_ratio": round(self.sortino_ratio(risk_free_rate), 3),
            "max_drawdown_pct": round(dd["max_drawdown"] * 100, 2),
            "max_drawdown_peak": dd["peak_date"].date().isoformat(),
            "max_drawdown_trough": dd["trough_date"].date().isoformat(),
            "historical_var_95_pct": round(self.value_at_risk(0.95, "historical") * 100, 2),
        }

    def print_summary(self, risk_free_rate: float = 0.0) -> None:
        s = self.summary(risk_free_rate)
        print(f"\n=== {s['name']} ({s['start_date']} to {s['end_date']}) ===")
        print(f"  Observations:          {s['num_observations']}")
        print(f"  Start / End price:     {s['start_price']} -> {s['end_price']}")
        print(f"  Total return:          {s['total_return_pct']}%")
        print(f"  Annualized return:     {s['annualized_return_pct']}%")
        print(f"  Annualized volatility: {s['annualized_volatility_pct']}%")
        print(f"  Sharpe ratio:          {s['sharpe_ratio']}")
        print(f"  Sortino ratio:         {s['sortino_ratio']}")
        print(f"  Max drawdown:          {s['max_drawdown_pct']}%  "
              f"(peak {s['max_drawdown_peak']} -> trough {s['max_drawdown_trough']})")
        print(f"  1-day historical VaR (95%): {s['historical_var_95_pct']}%")

    def plot(self, save_path: Optional[str] = None, ma_windows: Iterable[int] = (50, 200)):
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.prices.index, self.prices.values, label=self.name, linewidth=1.3)
        for w in ma_windows:
            ma = self.moving_average(w)
            ax.plot(ma.index, ma.values, label=f"{w}-day MA", linewidth=1)
        ax.set_title(f"{self.name} — Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved chart to {save_path}")
        return fig

    def plot_drawdown(self, save_path: Optional[str] = None):
        plt = _get_plt()
        dd = self.drawdown_series()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.4)
        ax.plot(dd.index, dd.values * 100, color="crimson", linewidth=1)
        ax.set_title(f"{self.name} — Drawdown (%)")
        ax.set_ylabel("Drawdown %")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved chart to {save_path}")
        return fig


class PortfolioComparison:
    """Compare several AssetAnalyzer instances side by side."""

    def __init__(self, analyzers: list[AssetAnalyzer]):
        if len(analyzers) < 2:
            raise ValueError("Provide at least two AssetAnalyzer instances to compare.")
        self.analyzers = analyzers

    def summary_table(self, risk_free_rate: float = 0.0) -> pd.DataFrame:
        rows = [a.summary(risk_free_rate) for a in self.analyzers]
        return pd.DataFrame(rows).set_index("name")

    def correlation_matrix(self) -> pd.DataFrame:
        returns = {a.name: a.returns for a in self.analyzers}
        return pd.DataFrame(returns).corr()

    def plot_cumulative_returns(self, save_path: Optional[str] = None):
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 5))
        for a in self.analyzers:
            cum = a.cumulative_returns()
            ax.plot(cum.index, cum.values, label=a.name, linewidth=1.3)
        ax.set_title("Growth of $1 — Cumulative Return Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved chart to {save_path}")
        return fig


def simulate_gbm_prices(s0: float = 100.0, mu: float = 0.08, sigma: float = 0.25,
                        days: int = 756, seed: Optional[int] = 7,
                        start_date: str = "2022-01-03") -> pd.Series:
    """Simulate a price path under Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS_PER_YEAR
    shocks = rng.normal(loc=(mu - 0.5 * sigma ** 2) * dt,
                        scale=sigma * np.sqrt(dt), size=days)
    log_prices = np.log(s0) + np.cumsum(shocks)
    prices = np.exp(log_prices)
    dates = pd.bdate_range(start=start_date, periods=days)
    return pd.Series(prices, index=dates, name="Simulated Asset")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze the historical return/risk profile of a financial asset."
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument("--ticker", type=str,
                        help="Single ticker to fetch via Yahoo Finance, e.g. AAPL")
    source.add_argument("--tickers", type=str, nargs="+",
                        help="Multiple tickers to compare, e.g. AAPL MSFT NVDA")
    source.add_argument("--csv", type=str,
                        help="Path to a local CSV file with a date column and a price column")
    p.add_argument("--start", type=str, default=None,
                   help="Start date, YYYY-MM-DD (for --ticker/--tickers)")
    p.add_argument("--end", type=str, default=None,
                   help="End date, YYYY-MM-DD (for --ticker/--tickers)")
    p.add_argument("--date-col", type=str, default="Date",
                   help="Date column name (for --csv)")
    p.add_argument("--price-col", type=str, default="Close",
                   help="Price column name (for --csv)")
    p.add_argument("--name", type=str, default=None,
                   help="Display name for the asset (for --csv)")
    p.add_argument("--risk-free", type=float, default=0.0,
                   help="Annual risk-free rate, e.g. 0.02 for 2%%")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directory to save chart PNGs into")
    p.add_argument("--no-plots", action="store_true", help="Skip generating chart files")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        analyzers = []
        for ticker in args.tickers:
            try:
                analyzers.append(AssetAnalyzer.from_yfinance(ticker, args.start, args.end))
            except Exception as e:
                print(f"Skipping {ticker}: {e}", file=sys.stderr)
        if len(analyzers) < 2:
            print("Need at least two successfully-fetched tickers to compare.", file=sys.stderr)
            return 1
        for analyzer in analyzers:
            analyzer.print_summary(args.risk_free)
        comparison = PortfolioComparison(analyzers)
        print("\nCorrelation matrix (daily returns):")
        print(comparison.correlation_matrix().round(3))
        if not args.no_plots:
            comparison.plot_cumulative_returns(
                str(out_dir / "comparison_cumulative_returns.png"))
        return 0

    if args.ticker:
        analyzer = AssetAnalyzer.from_yfinance(args.ticker, args.start, args.end)
    elif args.csv:
        analyzer = AssetAnalyzer.from_csv(args.csv, args.date_col, args.price_col, args.name)
    else:
        print("No data source given (--ticker / --tickers / --csv). "
              "Running a demo on simulated GBM price data instead.\n")
        analyzer = AssetAnalyzer(simulate_gbm_prices(), name="Simulated Asset (GBM demo)")

    analyzer.print_summary(args.risk_free)
    if not args.no_plots:
        analyzer.plot(str(out_dir / f"{analyzer.name.replace(' ', '_')}_price.png"))
        analyzer.plot_drawdown(str(out_dir / f"{analyzer.name.replace(' ', '_')}_drawdown.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
