# CryptoPrice Forecaster using TimesFM 3.0

A cryptocurrency price forecasting project powered by **Google's TimesFM 3.0**, a pretrained time-series foundation model.

This project retrieves real-time and historical cryptocurrency futures data from the Binance USD-M Futures API and uses TimesFM 3.0 to forecast future prices based on historical market information.

Unlike basic price-only forecasting, this project combines multiple market indicators, including **Price, Open Interest (OI), and Funding Rate**, to provide additional historical market context.

## Features

- **Dynamic Cryptocurrency Selection:** Enter any supported Binance USDT perpetual futures trading pair.
- **Binance API Integration:** Automatically downloads historical futures market data.
- **Multi-Indicator Dataset:** Combines price, trading volume, Open Interest, and Funding Rate.
- **TimesFM 3.0:** Uses Google's pretrained time-series foundation model for forecasting.
- **Custom Forecast Horizon:** Predict prices for a specified number of minutes, hours, or days.
- **Prediction Intervals:** Generates P10 and P90 quantile estimates alongside point forecasts.
- **Automatic Data Management:** Deletes the temporary merged CSV after a successful forecast while preserving the forecast output.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TimesFM 3.0 | Time-series forecasting |
| PyTorch | Model execution |
| NumPy | Numerical computation |
| Pandas | Data processing and merging |
| Requests | Binance API integration |
| Binance Futures API | Cryptocurrency market data |

## Data Collection

The project collects four major market features:

| Feature | Description |
|---|---|
| Time | Timestamp of each observation |
| Price | Closing price of the cryptocurrency |
| Open Interest | Total outstanding futures contracts |
| Funding Rate | Periodic funding rate of perpetual futures |
| Volume | Trading volume during the candle interval |

**Data configuration:**
- Price interval: 5 minutes
- Open Interest interval: 5 minutes
- Historical lookback: 29 days
- Model context: Last 2,048 observations
- Maximum forecast horizon: 7 days

The collected data is merged into a time-aligned dataset using Pandas.

## Project Workflow

1. Enter a cryptocurrency symbol (e.g., BTC, ETH, SOL).
2. The program validates the trading pair against Binance Futures.
3. Downloads historical price, volume, Open Interest, and Funding Rate.
4. Merges the indicators into a single time-series dataset.
5. Loads the pretrained TimesFM 3.0 model.
6. Processes historical Open Interest and Funding Rate as past-only covariates.
7. Generates future price predictions and quantile estimates.
8. Exports the forecast into a CSV file.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CryptoPrice-Forecaster-TimesFM.git
cd CryptoPrice-Forecaster-TimesFM
```

### 2. Install dependencies

```bash
python -m pip install numpy pandas requests
python -m pip install "timesfm[torch]"
```

### 3. Run the project

```bash
python predict.py
```

The pretrained TimesFM checkpoint is:

```text
google/timesfm-3.0-pytorch
```

The model may need to download its checkpoint the first time it runs.

## Usage

Run the program:

```bash
python predict.py
```

Example interaction:

```text
========== COIN INPUT ==========

Enter coin (e.g. BTC, ETH, SOL) or pair: BTC

Selected pair: BTCUSDT

Predict how many units ahead? 6

Unit (min/hour/day): hour
```

The model will download the required market data and generate a forecast for the requested horizon.

You can enter:
- BTC, ETH, SOL, etc.
- BTCUSDT, ETHUSDT, etc.
- BTC/USDT or BTC-USDT.

Supported forecast units:
- Minutes
- Hours
- Days

## Output

The program generates a forecast CSV file:

```text
BTCUSDT_forecast.csv
```

The output contains:

| Column | Description |
|---|---|
| time_ist | Forecast timestamp in Indian Standard Time |
| time_utc | Forecast timestamp in UTC |
| predicted_price | Predicted cryptocurrency price |
| p10 | 10th percentile forecast |
| p90 | 90th percentile forecast |

The P10 and P90 values provide a lower and upper quantile estimate from the model. They are not guaranteed price boundaries.

The intermediate merged dataset is automatically deleted after a successful forecast. If forecasting fails, the temporary dataset is retained for debugging.

## Important Notes

- Requires an active internet connection to retrieve Binance data and download the model checkpoint.
- Uses Binance USD-M USDT perpetual futures markets.
- Open Interest history is subject to Binance API availability and retention limits.
- The model runs on CPU in the current implementation.
- Forecasting uses historical Open Interest and Funding Rate as past-only covariates; it does not predict future indicator values.
- Missing or irregular five-minute observations can cause validation errors.

## Disclaimer

This project is developed for educational and research purposes.

Cryptocurrency markets are highly volatile. Model predictions are estimates based on historical patterns and should not be considered financial advice or guaranteed future prices.

**Do not use these predictions as the sole basis for trading or investment decisions.**

## Author

**Sampat Sheshagiri Naik**

AI & ML Engineering Student

GitHub: [@Sampat369](https://github.com/Sampat369)

---

If you find this project interesting, consider giving the repository a ⭐!
