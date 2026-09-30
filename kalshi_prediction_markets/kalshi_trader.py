"""
kalshi_trader.py
================
A paper-trading toolkit for Kalshi-style binary prediction markets.

Kalshi contracts trade in cents (1-99), represent a YES/NO outcome, and
settle at 0 or 100 when the underlying event resolves. This module gives
you everything EXCEPT the live API layer:

    - A synthetic market simulator (a bounded random walk in probability
      space that resolves to YES or NO at expiry -- a "Brownian bridge"
      toward a coin flip, in spirit)
    - Order / Fill / Position / Portfolio bookkeeping
    - A pluggable Strategy interface, with three example strategies
    - A backtester that runs a strategy over a simulated (or historical)
      price series and reports PnL, win rate, and drawdown
    - A KalshiClient class with the real API calls stubbed out --
      fill in `_request()` and auth when you're ready to go live

Nothing in here talks to the internet. Swap `KalshiClient`'s stubs for
real HTTP calls (see the TODOs) when you want to connect it to Kalshi's
actual REST API.

Usage:
    python kalshi_trader.py                     # run the built-in demo
    python kalshi_trader.py --steps 200 --fee 1  # tweak the simulation

As a library:
    from kalshi_trader import simulate_market, Backtester, ThresholdStrategy
    path = simulate_market(steps=180, resolves_yes=True)
    strat = ThresholdStrategy(buy_below=35, sell_above=65)
    result = Backtester(path, strat).run()
    print(result.summary())
"""

from __future__ import annotations

import argparse
import base64
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import quote, urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import numpy as np
import pandas as pd


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------
class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    ticker: str
    side: Side
    action: OrderAction
    quantity: int                  # number of contracts
    limit_price_cents: Optional[int] = None  # None = market order
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class Fill:
    order_id: str
    ticker: str
    side: Side
    action: OrderAction
    quantity: int
    price_cents: int                # actual execution price
    fee_cents: float
    timestamp: pd.Timestamp


@dataclass
class Position:
    ticker: str
    side: Side
    quantity: int = 0
    avg_price_cents: float = 0.0    # volume-weighted average entry price

    def market_value_cents(self, current_price_cents: int) -> float:
        """Current price, in cents, of the side this position holds."""
        px = current_price_cents if self.side == Side.YES else 100 - current_price_cents
        return self.quantity * px

    def unrealized_pnl_cents(self, current_price_cents: int) -> float:
        px = current_price_cents if self.side == Side.YES else 100 - current_price_cents
        return self.quantity * (px - self.avg_price_cents)


# ---------------------------------------------------------------------
# Market simulation
# ---------------------------------------------------------------------
def simulate_market(steps: int = 180, start_price: int = 50,
                     volatility: float = 6.0, resolves_yes: Optional[bool] = None,
                     seed: Optional[int] = None, freq: str = "h") -> pd.Series:
    """
    Simulate a YES-price path (in cents, 1-99) for a single Kalshi-style
    contract, from `steps` periods before expiry up to resolution.

    Modeled as a random walk on the log-odds (logit) scale -- this keeps
    the price naturally bounded in (0, 100) without clipping artifacts --
    with a drift term that pulls the path toward the eventual outcome as
    expiry approaches (a rough analogue of a Brownian bridge). The final
    value is snapped to 100 (YES) or 0 (NO).

    Parameters
    ----------
    steps : number of periods to simulate before resolution
    start_price : starting YES price in cents (1-99)
    volatility : logit-scale volatility per step (higher = choppier market)
    resolves_yes : True/False to force the outcome; None = random 50/50
    seed : RNG seed for reproducibility
    freq : pandas frequency string for the timestamp index (default hourly)
    """
    rng = np.random.default_rng(seed)
    if resolves_yes is None:
        resolves_yes = bool(rng.integers(0, 2))

    start_price = np.clip(start_price, 1, 99)
    logit = np.log(start_price / (100 - start_price))
    target_logit = 6.0 if resolves_yes else -6.0  # ~99.8% / ~0.2% in cents

    logits = [logit]
    for t in range(1, steps):
        time_left = 1 - t / steps
        pull = (target_logit - logits[-1]) * (0.05 / max(time_left, 0.02))
        shock = rng.normal(0, volatility / 10)
        logits.append(logits[-1] + pull * 0.02 + shock)

    prices = 100 / (1 + np.exp(-np.array(logits)))
    prices = np.clip(prices, 1, 99)
    prices[-1] = 100.0 if resolves_yes else 0.0
    prices = np.round(prices).astype(int)
    prices = np.clip(prices, 0, 100)

    index = pd.date_range(end=pd.Timestamp.now().floor(freq), periods=steps, freq=freq)
    return pd.Series(prices, index=index, name="yes_price_cents")


# ---------------------------------------------------------------------
# Portfolio / paper-trading exchange
# ---------------------------------------------------------------------
class Portfolio:
    """Tracks cash, positions, and fills for one or more Kalshi contracts."""

    def __init__(self, starting_cash_cents: int = 100_000):
        self.cash_cents = starting_cash_cents
        self.positions: dict[tuple[str, Side], Position] = {}
        self.fills: list[Fill] = []

    def _get_position(self, ticker: str, side: Side) -> Position:
        key = (ticker, side)
        if key not in self.positions:
            self.positions[key] = Position(ticker=ticker, side=side)
        return self.positions[key]

    def apply_fill(self, fill: Fill) -> None:
        pos = self._get_position(fill.ticker, fill.side)
        cost_cents = fill.quantity * fill.price_cents * (1 if fill.action == OrderAction.BUY else -1)

        if fill.action == OrderAction.BUY:
            new_qty = pos.quantity + fill.quantity
            if new_qty > 0:
                pos.avg_price_cents = (
                    (pos.avg_price_cents * pos.quantity + fill.price_cents * fill.quantity) / new_qty
                )
            pos.quantity = new_qty
        else:  # SELL
            pos.quantity -= fill.quantity
            if pos.quantity <= 0:
                pos.quantity = max(pos.quantity, 0)  # no naked shorting in this simple model

        self.cash_cents -= cost_cents
        self.cash_cents -= fill.fee_cents
        self.fills.append(fill)

    def mark_to_market_cents(self, current_price_cents: int) -> float:
        value = self.cash_cents
        for pos in self.positions.values():
            value += pos.market_value_cents(current_price_cents)
        return value

    def net_position(self, ticker: str) -> int:
        """YES contracts minus NO contracts held, for a quick net-exposure read."""
        yes_qty = self.positions.get((ticker, Side.YES), Position(ticker, Side.YES)).quantity
        no_qty = self.positions.get((ticker, Side.NO), Position(ticker, Side.NO)).quantity
        return yes_qty - no_qty


class PaperExchange:
    """
    Executes orders against a simulated market price with a configurable
    bid/ask spread and flat per-contract fee, and books the resulting
    fills into a Portfolio. Swap this out for a real order-routing layer
    when you connect the live API.
    """

    def __init__(self, portfolio: Portfolio, spread_cents: int = 1, fee_cents_per_contract: float = 0.5):
        self.portfolio = portfolio
        self.spread_cents = spread_cents
        self.fee_cents_per_contract = fee_cents_per_contract

    def execute(self, order: Order, current_yes_price: int, timestamp: pd.Timestamp) -> Fill:
        # Market orders cross the spread; limit orders are assumed filled
        # at their limit price if it's marketable (kept simple on purpose).
        mid = current_yes_price if order.side == Side.YES else 100 - current_yes_price
        if order.action == OrderAction.BUY:
            exec_price = mid + self.spread_cents // 2
        else:
            exec_price = mid - self.spread_cents // 2
        exec_price = int(np.clip(exec_price, 0, 100))

        if order.limit_price_cents is not None:
            if order.action == OrderAction.BUY and exec_price > order.limit_price_cents:
                exec_price = order.limit_price_cents
            elif order.action == OrderAction.SELL and exec_price < order.limit_price_cents:
                exec_price = order.limit_price_cents

        fee = self.fee_cents_per_contract * order.quantity
        fill = Fill(
            order_id=order.order_id, ticker=order.ticker, side=order.side,
            action=order.action, quantity=order.quantity, price_cents=exec_price,
            fee_cents=fee, timestamp=timestamp,
        )
        self.portfolio.apply_fill(fill)
        return fill


# ---------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------
class Strategy(ABC):
    """
    Subclass this and implement `generate_orders`. Called once per period
    with the price history so far and the current portfolio state; return
    a list of Orders to submit this period (usually 0 or 1).
    """

    @abstractmethod
    def generate_orders(self, ticker: str, price_history: pd.Series,
                         portfolio: Portfolio) -> list[Order]:
        ...


class ThresholdStrategy(Strategy):
    """Buy YES when it looks cheap, buy NO (i.e. bet against YES) when YES looks rich."""

    def __init__(self, buy_below: int = 35, sell_above: int = 65, quantity: int = 10):
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.quantity = quantity

    def generate_orders(self, ticker, price_history, portfolio):
        price = int(price_history.iloc[-1])
        orders = []
        if price < self.buy_below and portfolio.net_position(ticker) <= 0:
            orders.append(Order(ticker, Side.YES, OrderAction.BUY, self.quantity))
        elif price > self.sell_above and portfolio.net_position(ticker) >= 0:
            orders.append(Order(ticker, Side.NO, OrderAction.BUY, self.quantity))
        return orders


class MomentumStrategy(Strategy):
    """Buy the side the price has been trending toward over a lookback window."""

    def __init__(self, lookback: int = 10, min_move_cents: int = 5, quantity: int = 10):
        self.lookback = lookback
        self.min_move_cents = min_move_cents
        self.quantity = quantity

    def generate_orders(self, ticker, price_history, portfolio):
        if len(price_history) <= self.lookback:
            return []
        change = price_history.iloc[-1] - price_history.iloc[-1 - self.lookback]
        orders = []
        if change >= self.min_move_cents and portfolio.net_position(ticker) <= 0:
            orders.append(Order(ticker, Side.YES, OrderAction.BUY, self.quantity))
        elif change <= -self.min_move_cents and portfolio.net_position(ticker) >= 0:
            orders.append(Order(ticker, Side.NO, OrderAction.BUY, self.quantity))
        return orders


class MeanReversionStrategy(Strategy):
    """Fade large, fast moves away from a rolling average."""

    def __init__(self, window: int = 20, z_threshold: float = 1.5, quantity: int = 10):
        self.window = window
        self.z_threshold = z_threshold
        self.quantity = quantity

    def generate_orders(self, ticker, price_history, portfolio):
        if len(price_history) < self.window:
            return []
        recent = price_history.iloc[-self.window:]
        mean, std = recent.mean(), recent.std()
        if std == 0 or np.isnan(std):
            return []
        z = (price_history.iloc[-1] - mean) / std
        orders = []
        if z < -self.z_threshold and portfolio.net_position(ticker) <= 0:
            orders.append(Order(ticker, Side.YES, OrderAction.BUY, self.quantity))
        elif z > self.z_threshold and portfolio.net_position(ticker) >= 0:
            orders.append(Order(ticker, Side.NO, OrderAction.BUY, self.quantity))
        return orders


# ---------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------
@dataclass
class BacktestResult:
    ticker: str
    equity_curve: pd.Series          # portfolio value (cents) at each step
    fills: list[Fill]
    starting_cash_cents: int
    ending_cash_cents: float
    resolved_yes: bool

    def summary(self) -> dict:
        pnl = self.equity_curve.iloc[-1] - self.starting_cash_cents
        returns = self.equity_curve.pct_change().dropna()
        max_dd = ((self.equity_curve / self.equity_curve.cummax()) - 1).min()
        return {
            "ticker": self.ticker,
            "resolved_yes": self.resolved_yes,
            "num_fills": len(self.fills),
            "starting_cash_cents": self.starting_cash_cents,
            "ending_value_cents": round(self.equity_curve.iloc[-1], 1),
            "pnl_cents": round(pnl, 1),
            "pnl_pct": round(pnl / self.starting_cash_cents * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2) if not np.isnan(max_dd) else 0.0,
            "return_volatility": round(returns.std(), 5) if len(returns) else 0.0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== Backtest: {s['ticker']} (resolved {'YES' if s['resolved_yes'] else 'NO'}) ===")
        print(f"  Fills:              {s['num_fills']}")
        print(f"  Starting cash:      {s['starting_cash_cents']}¢")
        print(f"  Ending value:       {s['ending_value_cents']}¢")
        print(f"  PnL:                {s['pnl_cents']}¢  ({s['pnl_pct']}%)")
        print(f"  Max drawdown:       {s['max_drawdown_pct']}%")

    def plot(self, save_path: Optional[str] = None):
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.equity_curve.index, self.equity_curve.values, linewidth=1.3)
        ax.axhline(self.starting_cash_cents, color="gray", linestyle="--", linewidth=1, label="Starting cash")
        ax.set_title(f"{self.ticker} — Paper Trading Equity Curve")
        ax.set_ylabel("Portfolio value (¢)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved chart to {save_path}")
        return fig


class Backtester:
    def __init__(self, price_history: pd.Series, strategy: Strategy, ticker: str = "DEMO-MKT",
                 starting_cash_cents: int = 100_000, spread_cents: int = 1,
                 fee_cents_per_contract: float = 0.5):
        self.price_history = price_history
        self.strategy = strategy
        self.ticker = ticker
        self.starting_cash_cents = starting_cash_cents
        self.portfolio = Portfolio(starting_cash_cents)
        self.exchange = PaperExchange(self.portfolio, spread_cents, fee_cents_per_contract)

    def run(self) -> BacktestResult:
        equity = []
        for i in range(1, len(self.price_history) + 1):
            history_so_far = self.price_history.iloc[:i]
            current_price = int(history_so_far.iloc[-1])
            ts = history_so_far.index[-1]

            orders = self.strategy.generate_orders(self.ticker, history_so_far, self.portfolio)
            for order in orders:
                self.exchange.execute(order, current_price, ts)

            equity.append(self.portfolio.mark_to_market_cents(current_price))

        equity_curve = pd.Series(equity, index=self.price_history.index, name="equity_cents")
        resolved_yes = self.price_history.iloc[-1] >= 50

        return BacktestResult(
            ticker=self.ticker,
            equity_curve=equity_curve,
            fills=self.portfolio.fills,
            starting_cash_cents=self.starting_cash_cents,
            ending_cash_cents=self.portfolio.cash_cents,
            resolved_yes=bool(resolved_yes),
        )


# ---------------------------------------------------------------------
# Kalshi API client
# ---------------------------------------------------------------------
class KalshiClient:
    """
    Authenticated Kalshi REST client. Defaults to the demo environment;
    demo and production credentials are separate.
    """

    BASE_URLS = {
        "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
        "production": "https://external-api.kalshi.com/trade-api/v2",
    }

    def __init__(self, api_key_id: Optional[str] = None, private_key_path: Optional[str] = None,
                 environment: str = "demo", allow_live_trading: bool = False,
                 timeout: float = 10.0, session: Optional[requests.Session] = None):
        if environment not in self.BASE_URLS:
            raise ValueError("environment must be 'demo' or 'production'")

        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not self.api_key_id or not self.private_key_path:
            raise ValueError(
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH, "
                "or pass api_key_id and private_key_path."
            )

        key_path = os.path.expanduser(self.private_key_path)
        with open(key_path, "rb") as key_file:
            self.private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        if not isinstance(self.private_key, (rsa.RSAPrivateKey, Ed25519PrivateKey)):
            raise TypeError("Kalshi API keys must use RSA or Ed25519.")

        self.environment = environment
        self.base_url = self.BASE_URLS[environment]
        self.allow_live_trading = allow_live_trading
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        method = method.upper()
        if self.environment == "production" and method in {"POST", "PUT", "PATCH", "DELETE"}:
            if not self.allow_live_trading:
                raise RuntimeError(
                    "Production writes are disabled. Set allow_live_trading=True "
                    "only after verifying your production configuration."
                )

        url = f"{self.base_url}{path}"
        timestamp = str(int(time.time() * 1000))
        signed_path = urlparse(url).path
        message = f"{timestamp}{method}{signed_path}".encode("utf-8")

        if isinstance(self.private_key, Ed25519PrivateKey):
            signature = self.private_key.sign(message)
        else:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        }
        timeout = kwargs.pop("timeout", self.timeout)
        response = self.session.request(
            method, url, headers=headers, timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response.json() if response.content else None

    def get_markets(self, series_ticker: Optional[str] = None, status: Optional[str] = None):
        params = {key: value for key, value in {
            "series_ticker": series_ticker, "status": status,
        }.items() if value is not None}
        return self._request("GET", "/markets", params=params)

    def get_orderbook(self, ticker: str):
        return self._request("GET", f"/markets/{quote(ticker, safe='')}/orderbook")

    def get_positions(self):
        return self._request("GET", "/portfolio/positions")

    def get_api_keys(self):
        return self._request("GET", "/api_keys")

    def place_order(self, ticker: str, side: Side, action: OrderAction,
                     quantity: int, limit_price_cents: Optional[int] = None,
                     client_order_id: Optional[str] = None):
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if limit_price_cents is None or not 1 <= limit_price_cents <= 99:
            raise ValueError("limit_price_cents must be between 1 and 99")

        book_side = {
            (Side.YES, OrderAction.BUY): "bid",
            (Side.YES, OrderAction.SELL): "ask",
            (Side.NO, OrderAction.BUY): "ask",
            (Side.NO, OrderAction.SELL): "bid",
        }[(side, action)]
        yes_price_cents = (
            limit_price_cents if side == Side.YES else 100 - limit_price_cents
        )
        payload = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": book_side,
            "count": str(quantity),
            "price": f"{yes_price_cents / 100:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        return self._request("POST", "/portfolio/events/orders", json=payload)

    def cancel_order(self, order_id: str, market_ticker: str):
        return self._request(
            "DELETE",
            f"/portfolio/events/orders/{quote(order_id, safe='')}",
            params={"market_ticker": market_ticker},
        )


# ---------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-trade a simulated Kalshi-style prediction market.")
    p.add_argument("--steps", type=int, default=180, help="Number of periods before resolution")
    p.add_argument("--start-price", type=int, default=50, help="Starting YES price in cents (1-99)")
    p.add_argument("--volatility", type=float, default=6.0, help="Logit-scale volatility per step")
    p.add_argument("--resolves", type=str, choices=["yes", "no", "random"], default="random")
    p.add_argument("--strategy", type=str, choices=["threshold", "momentum", "meanrev"], default="threshold")
    p.add_argument("--fee", type=float, default=0.5, help="Fee in cents per contract")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    resolves_yes = {"yes": True, "no": False, "random": None}[args.resolves]

    path = simulate_market(
        steps=args.steps, start_price=args.start_price, volatility=args.volatility,
        resolves_yes=resolves_yes, seed=args.seed,
    )

    strategy_map = {
        "threshold": ThresholdStrategy(),
        "momentum": MomentumStrategy(),
        "meanrev": MeanReversionStrategy(),
    }
    strategy = strategy_map[args.strategy]

    bt = Backtester(path, strategy, ticker=f"DEMO-{args.strategy.upper()}", fee_cents_per_contract=args.fee)
    result = bt.run()
    result.print_summary()

    if not args.no_plots:
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.plot(str(out_dir / f"{result.ticker}_equity.png"))

        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(path.index, path.values, linewidth=1.3, color="darkorange")
        ax.axhline(50, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{result.ticker} — Simulated YES Price (¢)")
        ax.set_ylabel("YES price (¢)")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{result.ticker}_price.png"), dpi=150)
        print(f"Saved chart to {out_dir / f'{result.ticker}_price.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
