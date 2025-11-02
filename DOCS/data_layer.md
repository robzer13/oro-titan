# OroTitan Data Layer

## 1. Pipeline officiel

### 1. Validate existing prices

   python .\scripts\validate_prices.py .\prices.parquet

   - if it prints "ERROR: all `close` values are NaN" and exits with code 2 -> go to step 2

### 2. Try to fix dirty prices

   python .\scripts\fix_prices.py .\prices.parquet .\prices_fixed.parquet

   python .\scripts\validate_prices.py .\prices_fixed.parquet

   - if it prints "FATAL: cannot fix prices (close still empty)" -> go to step 3

### 3. Rebuild from yfinance (requires network)

   python .\scripts\build_prices_yf.py .\tickers.txt .\prices_built.parquet 2015-01-01

   python .\scripts\validate_prices.py .\prices_built.parquet

### 4. Add factors

   python .\scripts\add_factors.py .\prices_built.parquet .\prices_with_factors.parquet

### 5. Run backtest

   python .\backtest_runner.py --prices .\prices_with_factors.parquet --config .\configs\scoring.yml --date_start 2018-01-01 --curve_out .\out\curve.csv --weights_out .\out\weights_log.csv --picks_out .\out\picks_log.csv


## 2. Smoke tests

### 2.1 Synthetic smoke

Use when code was modified.

   python .\scripts\smoke_pipeline.py

This uses only synthetic data and must print:

   [smoke] PIPELINE OK

If it fails -> fix the Python scripts (validate/fix/build_yf/add_factors/backtest_runner).


### 2.2 Real-data smoke

Use when tickers.txt changed or when we added markets.

   python .\scripts\smoke_pipeline_real.py

This must print:

   [smoke-real] PIPELINE REAL OK

If it fails -> check .\tickers.txt and rebuild from yfinance.


## 3. Common errors

- "[validate] ERROR: all `close` values are NaN."

  Meaning: your parquet is empty or unusable.

  Action: run build_prices_yf.py or use the synthetic generator.


- "FATAL: cannot fix prices (close still empty)"

  Meaning: fix_prices.py could not salvage close from adj_close.

  Action: rebuild from yfinance.


- "[runner] ERROR: universe too small (<3 tickers)."

  Meaning: after applying options.universe.include from configs\scoring.yml (usually ["*.PA","*.AS"]), there are not enough tickers.

  Action: add more tickers in .\tickers.txt OR relax universe.include in configs\scoring.yml.


- "[runner] ERROR: insufficient coverage (<500 rows or <18 months)."

  Meaning: data exists but is too short.

  Action: rebuild from yfinance with an earlier start date (example: 2014-01-01).


## 4. Notes

- All scripts print ASCII-only to support Windows consoles (cp1252).

- CLI dates have priority over dates from the YAML: --date_start/--date_end > --start/--end > options.date_start/date_end.

- If validation fails, run the yfinance rebuild.

