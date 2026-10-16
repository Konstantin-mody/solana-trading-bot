# Solana Trading Bot

An algorithmic trading bot for SOL/USD (plus early experiments with BTC and the S&P 500), built over about three months in summer 2026 right after my Abitur. I wrote it together with Claude Code in Python.

## What it does

- Detects the market regime (bear, sideways, bull, extreme bull) using EMA, DI/ADX and moving averages
- Trades each regime with its own rules, e.g. shorts in bear markets, RSI and trend entries in bull markets
- Uses ATR based stop losses, take profits and trailing stops plus leverage depending on the regime
- Runs backtests over roughly five years of Binance data (2021 to 2026) with a cost model for fees, slippage and funding
- Has a paper trading mode that runs the strategy live without real money

## Strategies I tested

RSI + EMA, MACD, Bollinger Bands, Larry Connors RSI(2), RSI divergence, support and resistance levels and Fibonacci setups. I also built separate versions for different timeframes (15m, 1h, 4h, 1d, 1w) and compared them.

## Why it's paused

The backtests looked really good at first, but over time I found calculation errors in the logic (fees, position sizing, averaging) that made the results look better than they were. The 15 minute version also turned out to be impossible to make profitable because the trading costs ate the whole edge. Once uni started I paused the project.

Honestly that was the most valuable part. I learned how easy it is to fool yourself with a backtest and how important it is to question your own numbers.

## Structure

```
sol_bot/          1h bot (main version)
sol_bot_15m/      15 minute bot (discarded)
sol_bot_4h/       4h engine
sol_bot_1d/       daily engine
sol_bot_1w/       weekly engine
cost_model.py     shared fees, slippage and funding model
run_backtest*.py  backtest runners for each timeframe
paper_bot*.py     paper trading
research/         analysis and sanity tests
archive/          older experiments (first BTC bot, Binance testnet bot)
docs/             strategy notes, results and roadmap (mostly in German)
```

## Run it

```bash
pip3 install -r requirements.txt
python3 run_backtest.py --data binance
```

For the testnet bot in `archive/live_bot.py` copy `.env.example` to `.env` and add your own Binance testnet keys.

## Disclaimer

This is a learning project, not financial advice. Don't trade real money with it.
