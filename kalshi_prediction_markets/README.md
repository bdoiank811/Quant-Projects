# Kalshi Prediction Markets

A Python toolkit for experimenting with Kalshi-style binary markets. It combines a synthetic market simulator, a paper-trading ledger, strategy backtests, and a demo-first client for Kalshi's authenticated REST API.

This project is for research and testing. Simulated performance is not a forecast of future results, and the demo environment uses separate credentials and balances from production.

## Capabilities

- **Market simulation:** Generates a reproducible YES-price path from 1 to 99 cents, then resolves it to 0 or 100. The path is a random walk in log-odds space with a drift toward the selected outcome.
- **Paper trading:** Tracks cash, YES/NO positions, weighted-average entry prices, fills, fees, and marked-to-market portfolio value. `PaperExchange` simulates a configurable spread and per-contract fee; its limit fills are deliberately simplified.
- **Strategies:** Includes threshold, momentum, and mean-reversion examples behind a common `Strategy` interface.
- **Backtesting:** Runs a strategy over a `pandas.Series` of prices and reports fills, ending value, PnL, drawdown, and return volatility. Historical price data must be supplied by the caller; the project does not download it.
- **Kalshi REST client:** Signs requests with RSA-PSS or Ed25519 keys, defaults to the demo environment, and supports market, orderbook, position, and API-key reads plus limit-order placement and cancellation.

## Install

From this directory, install the dependencies into your Python environment:

```bash
python -m pip install matplotlib numpy pandas requests cryptography
```

## Run a Backtest

Run the built-in simulation and save equity and price charts:

```bash
python kalshi_trader.py
```

Run a reproducible 200-step simulation without writing charts:

```bash
python kalshi_trader.py --steps 200 --fee 1 --seed 42 --no-plots
```

Select the resolution with `--resolves yes|no|random` and the strategy with `--strategy threshold|momentum|meanrev`. Use `python kalshi_trader.py --help` to see all options. By default, charts are saved in the current directory; `--output-dir` changes the destination.

The toolkit can also be used as a library:

```python
from kalshi_trader import Backtester, ThresholdStrategy, simulate_market

prices = simulate_market(steps=180, resolves_yes=True, seed=42)
result = Backtester(
	prices,
	ThresholdStrategy(buy_below=35, sell_above=65),
).run()
print(result.summary())
```

## Kalshi Demo API

Create and use an account with Kalshi's [demo account tutorial](https://help.kalshi.com/en/articles/13823775-creating-and-using-a-demo-account), then create an API key from that demo account. Demo credentials are not interchangeable with production credentials. The key-management page may provide an RSA key; the client also supports Ed25519 PEM keys.

Keep the private key outside this repository. Set the demo key ID and private-key path in your shell; do not paste the key into source code, commit it, or share it in chat:

```bash
export KALSHI_API_KEY_ID="your-demo-key-id"
export KALSHI_PRIVATE_KEY_PATH="$HOME/.config/kalshi/demo-private-key.pem"
```

The client defaults to the recommended demo REST root. Kalshi also supports the alternate REST host and WebSocket hosts below. WebSocket streaming is not implemented in this project.

| Surface | Recommended demo endpoint | Also supported |
| --- | --- | --- |
| REST Trade API | `https://external-api.demo.kalshi.co/trade-api/v2` | `https://demo-api.kalshi.co/trade-api/v2` |
| WebSocket API | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` | `wss://demo-api.kalshi.co/trade-api/ws/v2` |

Try an authenticated read-only request:

```bash
python -c 'from kalshi_trader import KalshiClient; print(KalshiClient().get_markets(status="open"))'
```

The client exposes `get_markets()`, `get_orderbook(ticker)`, `get_positions()`, and `get_api_keys()`. Order placement requires a positive quantity and a limit price from 1 to 99 cents. Replace the example ticker with an open ticker returned by `get_markets()`:

```python
from kalshi_trader import KalshiClient, OrderAction, Side

client = KalshiClient()  # demo is the default
order = client.place_order(
	ticker="YOUR_DEMO_MARKET_TICKER",
	side=Side.YES,
	action=OrderAction.BUY,
	quantity=1,
	limit_price_cents=10,
)
print(order)
```

Cancel with both identifiers:

```python
client.cancel_order(order_id=order["order_id"], market_ticker="YOUR_DEMO_MARKET_TICKER")
```

The client signs the timestamp, HTTP method, and URL path (excluding query parameters) for each request. Demo order calls use demo funds. Production reads require production credentials; production order placement and cancellation are additionally blocked unless the client is constructed with both `environment="production"` and `allow_live_trading=True`. The simulator and backtester do not submit orders to Kalshi.

## Project Files

- `kalshi_trader.py` contains the simulator, ledger, strategies, backtester, REST client, and CLI.
- `README.md` documents setup and usage.
# Kalshi Prediction Markets

A Python paper-trading toolkit for Kalshi-style binary prediction markets. It simulates YES-price paths, tracks orders and positions, and backtests threshold, momentum, and mean-reversion strategies.

The `KalshiClient` connects to Kalshi's demo environment by default. Demo credentials are separate from production credentials. Production writes are blocked unless explicitly enabled in the client constructor.

## Setup

From this directory, install the dependencies:

```bash
python -m pip install matplotlib numpy pandas requests cryptography
```

## Run the demo

```bash
python kalshi_trader.py
```

Run a deterministic 200-step simulation without generating plots:

```bash
python kalshi_trader.py --steps 200 --fee 1 --seed 42 --no-plots
```

Choose the outcome and strategy with `--resolves yes|no|random` and `--strategy threshold|momentum|meanrev`. Use `--help` to see all options.

By default, chart images are written to the current directory. Set `--output-dir` to choose another location, or pass `--no-plots` to disable chart generation.

## Kalshi Demo API

Create and use a demo account with Kalshi's [step-by-step tutorial](https://help.kalshi.com/en/articles/13823775-creating-and-using-a-demo-account). Demo credentials and balances are separate from production. Create a demo API key, then store its key ID and private-key file path in your shell environment. Never put the private key in this project or commit it to Git.

The REST client uses Kalshi's recommended demo API root. Kalshi also supports an alternate REST host and separate WebSocket hosts; WebSocket streaming is not implemented in this client yet.

| Surface | Recommended demo endpoint | Also supported |
| --- | --- | --- |
| REST Trade API | `https://external-api.demo.kalshi.co/trade-api/v2` | `https://demo-api.kalshi.co/trade-api/v2` |
| WebSocket API | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` | `wss://demo-api.kalshi.co/trade-api/ws/v2` |

The WebSocket endpoints are for a future streaming client; the current `KalshiClient` uses REST only.

```bash
export KALSHI_API_KEY_ID="your-demo-key-id"
export KALSHI_PRIVATE_KEY_PATH="$HOME/.config/kalshi/demo-private-key.pem"
```

The client defaults to demo and uses the demo-specific credentials and API host. Try an authenticated, read-only request:

```bash
python -c 'from kalshi_trader import KalshiClient; print(KalshiClient().get_markets(status="open"))'
```

Demo order example (uses fake demo funds):

```python
from kalshi_trader import KalshiClient, OrderAction, Side

client = KalshiClient()
result = client.place_order(
	ticker="YOUR_DEMO_MARKET_TICKER",
	side=Side.YES,
	action=OrderAction.BUY,
	quantity=1,
	limit_price_cents=10,
)
print(result)
```

To cancel, pass both the returned order ID and its market ticker to `cancel_order`. Production requires `environment="production"`; production order placement and cancellation additionally require `allow_live_trading=True`.
