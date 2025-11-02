# oro_version.py
# Central version and QA thresholds for OroTitan data layer.

VERSION = "OroTitan_Data_v1.0"

# Synthetic path (make_synth_prices.py -> factors -> backtest)
MIN_ROWS_SYNTH = 1500

# Real path (build_prices_yf.py -> factors -> backtest)
MIN_ROWS_REAL = 2000

COMMENT = "Central version and QA thresholds for data layer."
