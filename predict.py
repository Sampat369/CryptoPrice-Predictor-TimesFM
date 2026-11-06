"""Download Binance USD-M futures data and forecast a user-entered coin with TimesFM 3.0.

Type a base coin (BTC) or full pair (BTCUSDT). The merged input CSV is
temporary and is deleted after a successful forecast; the forecast CSV is kept.
"""

from __future__ import annotations

import math
import re
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# TimesFM 3.0 (installed by: python -m pip install "timesfm[torch]")
from timesfm3 import ModelConfig, TimesFM3Evaluator


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

BASE_URL = "https://fapi.binance.com"
BAR_INTERVAL = "5m"          # Price candle interval
BAR_MINUTES = 5
OI_PERIOD = "5m"             # Must match the price interval
LOOKBACK_DAYS = 29           # Open Interest endpoint is limited to recent history
CONTEXT_POINTS = 2048        # Last 2048 observations (about 7 days at 5-minute bars)
REQUEST_TIMEOUT = 30
DELETE_TEMP_CSV_AFTER_SUCCESS = True



# ---------------------------------------------------------------------
# SELECT COIN AND FORECAST HORIZON
# ---------------------------------------------------------------------

def get_coin_pair() -> str:
    """Ask the user to type a coin or Binance USD-M Futures pair.

    Examples: BTC -> BTCUSDT, ETHUSDT -> ETHUSDT, BTC/USDT -> BTCUSDT.
    """
    print("\n========== COIN INPUT ==========")
    entered = input(
        "Enter coin (e.g. BTC, ETH, SOL) or pair (e.g. BTCUSDT): "
    ).strip().upper()

    # Allow common pair formats such as BTC/USDT or BTC-USDT.
    symbol = entered.replace("/", "").replace("-", "").replace("_", "")

    if not symbol:
        raise ValueError("Coin/pair cannot be empty.")

    # If the user types only BTC, convert it to BTCUSDT.
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    if not re.fullmatch(r"[A-Z0-9]{3,20}", symbol):
        raise ValueError(
            "Invalid coin/pair. Example inputs: BTC, ETH, BTCUSDT, BTC/USDT."
        )

    # Confirm that this is an active USDⓈ-M perpetual futures pair.
    response = requests.get(
        BASE_URL + "/fapi/v1/exchangeInfo",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    exchange_info = response.json()

    matching = [
        item for item in exchange_info.get("symbols", [])
        if item.get("symbol") == symbol
        and item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    ]

    if not matching:
        raise ValueError(
            f"{symbol} is not an active Binance USD-M USDT perpetual pair. "
            "Check the coin name and try again."
        )

    print(f"Selected pair: {symbol}")
    return symbol


def get_forecast_horizon() -> tuple[int, str, int]:
    amount_text = input("Predict how many units ahead? ").strip()
    if not amount_text.isdigit() or int(amount_text) <= 0:
        raise ValueError("Forecast amount must be a positive whole number.")
    amount = int(amount_text)

    unit = input("Unit (min/hour/day): ").strip().lower()
    multipliers = {
        "min": 1,
        "minute": 1,
        "minutes": 1,
        "hour": 60,
        "hours": 60,
        "day": 1440,
        "days": 1440,
    }
    if unit not in multipliers:
        raise ValueError("Use min, hour, or day.")

    requested_minutes = amount * multipliers[unit]
    # Data is sampled every five minutes, so round up to a whole bar.
    horizon = math.ceil(requested_minutes / BAR_MINUTES)
    actual_minutes = horizon * BAR_MINUTES

    # Keep a practical limit for a local forecasting run.
    if horizon > 2016:  # 7 days at 5-minute frequency
        raise ValueError("Please request a horizon of 7 days or less.")

    return horizon, unit, actual_minutes


# ---------------------------------------------------------------------
# BINANCE REQUEST + PAGINATION
# ---------------------------------------------------------------------

def fetch_data(endpoint: str, params: dict, time_key, limit: int) -> list:
    all_data = []
    start = int(params["startTime"])
    end = int(params["endTime"])

    while start <= end:
        request_params = {
            **params,
            "startTime": start,
            "endTime": end,
            "limit": limit,
        }

        response = requests.get(
            BASE_URL + endpoint,
            params=request_params,
            timeout=REQUEST_TIMEOUT,
        )

        if not response.ok:
            raise RuntimeError(
                f"Binance API error {response.status_code} for {endpoint}: "
                f"{response.text[:500]}"
            )

        data = response.json()
        if isinstance(data, dict):
            raise RuntimeError(f"Binance API returned an error: {data}")
        if not data:
            break

        all_data.extend(data)

        try:
            last_time = max(int(row[time_key]) for row in data)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read timestamp field {time_key!r} from {endpoint}."
            ) from exc

        if last_time < start:
            break

        next_start = last_time + 1
        if next_start <= start:
            break
        start = next_start

        print(f"Downloaded {len(all_data):,} records from {endpoint}")
        time_module.sleep(0.15)

        if len(data) < limit:
            break

    return all_data


# ---------------------------------------------------------------------
# DOWNLOAD PRICE + VOLUME
# ---------------------------------------------------------------------

def get_price_volume(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = fetch_data(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": BAR_INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        time_key=0,
        limit=1500,
    )

    if not data:
        raise ValueError(f"No price candles returned for {symbol}.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["close_time"], errors="coerce"),
        unit="ms",
        utc=True,
    )
    df["price"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Exclude the currently forming candle (its close_time is still in the future).
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["time"] < now_utc].copy()
    df = df[["time", "price", "volume"]].dropna()
    df = df.sort_values("time").drop_duplicates("time", keep="last")

    if df.empty:
        raise ValueError("No completed price candles are available.")
    return df


# ---------------------------------------------------------------------
# DOWNLOAD OPEN INTEREST
# ---------------------------------------------------------------------

def get_open_interest(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = fetch_data(
        "/futures/data/openInterestHist",
        {
            "symbol": symbol,
            "period": OI_PERIOD,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        time_key="timestamp",
        limit=500,
    )

    if not data:
        raise ValueError(
            "No Open Interest data returned. Binance only provides recent "
            "historical Open Interest statistics for this endpoint."
        )

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
    )
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df = df[["time", "oi"]].dropna()
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    return df


# ---------------------------------------------------------------------
# DOWNLOAD FUNDING RATE
# ---------------------------------------------------------------------

def get_funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = fetch_data(
        "/fapi/v1/fundingRate",
        {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        time_key="fundingTime",
        limit=1000,
    )

    if not data:
        raise ValueError(f"No funding-rate data returned for {symbol}.")

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        pd.to_numeric(df["fundingTime"], errors="coerce"),
        unit="ms",
        utc=True,
    )
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["time", "funding_rate"]].dropna()
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    return df


# ---------------------------------------------------------------------
# MERGE AND SAVE TEMPORARY DATASET
# ---------------------------------------------------------------------

def create_temporary_dataset(symbol: str, csv_path: Path) -> None:
    end_time = datetime.now(timezone.utc)
    # A small margin avoids requesting data just outside Binance's 30-day OI window.
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print("\nDownloading price and volume...")
    price_df = get_price_volume(symbol, start_ms, end_ms)

    print("\nDownloading Open Interest...")
    oi_df = get_open_interest(symbol, start_ms, end_ms)

    print("\nDownloading funding rate...")
    funding_start_ms = int(
        (start_time - timedelta(days=2)).timestamp() * 1000
    )
    funding_df = get_funding_rate(symbol, funding_start_ms, end_ms)

    # merge_asof(direction='backward') only attaches observations at or before
    # each price timestamp; it does not leak future indicator values.
    merged = pd.merge_asof(
        price_df.sort_values("time"),
        oi_df.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )

    merged = pd.merge_asof(
        merged.sort_values("time"),
        funding_df.sort_values("time"),
        on="time",
        direction="backward",
    )

    # Carry forward only recent OI readings; funding rate stays valid until
    # the next funding event is published/recorded.
    merged["oi"] = merged["oi"].ffill(limit=2)
    merged["funding_rate"] = merged["funding_rate"].ffill()
    merged = merged[["time", "price", "oi", "funding_rate", "volume"]]
    merged = merged.dropna(subset=["time", "price", "oi", "funding_rate"])
    merged = merged.sort_values("time").drop_duplicates("time", keep="last")

    if len(merged) < 128:
        raise ValueError(
            f"Only {len(merged)} usable rows remain after merging. "
            "Check whether the pair has enough price and indicator history."
        )

    merged.to_csv(csv_path, index=False)
    print(f"\nTemporary dataset created: {csv_path}")
    print(f"Rows: {len(merged):,}")
    print("Missing values:")
    print(merged.isna().sum().to_string())


# ---------------------------------------------------------------------
# LOAD + VALIDATE TEMPORARY CSV
# ---------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["time", "price", "oi", "funding_rate", "volume"]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {missing_columns}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ["price", "oi", "funding_rate", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    df = df.reset_index(drop=True)

    if len(df) < 128:
        raise ValueError("Not enough clean rows for forecasting (need at least 128).")

    # TimesFM expects regularly spaced observations. Do not silently train on
    # irregular timestamps, because that would distort the requested horizon.
    differences = df["time"].diff().dropna()
    expected_step = pd.Timedelta(minutes=BAR_MINUTES)
    if not differences.eq(expected_step).all():
        bad_count = int((differences != expected_step).sum())
        raise ValueError(
            f"The merged data has {bad_count} missing/irregular 5-minute steps. "
            "Keep the temporary CSV for inspection and check the OI/price merge."
        )

    return df.tail(CONTEXT_POINTS).copy()


# ---------------------------------------------------------------------
# LOAD TIMESFM 3.0
# ---------------------------------------------------------------------

def load_model() -> TimesFM3Evaluator:
    print("\nLoading TimesFM 3.0 on CPU...")
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=1,
        device="cpu",
    )
    return TimesFM3Evaluator(config)


# ---------------------------------------------------------------------
# FORECAST PRICE USING PAST OI + FUNDING RATE AS COVARIATES
# ---------------------------------------------------------------------

def predict_price(
    df: pd.DataFrame,
    model: TimesFM3Evaluator,
    horizon: int,
):
    prices = df["price"].to_numpy(dtype=np.float32)

    # Open Interest is usually a positive, scale-sensitive quantity.
    oi = np.log1p(np.maximum(df["oi"].to_numpy(dtype=np.float32), 0.0))
    funding = df["funding_rate"].to_numpy(dtype=np.float32)

    # Two historical covariate channels: (2, context_length).
    covariates = np.vstack([oi, funding]).astype(np.float32)

    # Standardize each covariate using only the observed context.
    means = covariates.mean(axis=1, keepdims=True)
    stds = covariates.std(axis=1, keepdims=True)
    stds[stds < 1e-6] = 1.0
    covariates = (covariates - means) / stds

    print("Generating forecast...")
    outputs = list(
        model.predict_batch(
            contexts=[prices],
            horizon=horizon,
            past_only_covariates=[covariates],
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )

    if not outputs:
        raise RuntimeError("TimesFM did not return a forecast.")

    result = outputs[0]
    forecast = np.asarray(result.forecast, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(result.quantiles, dtype=np.float64)

    if len(forecast) != horizon:
        raise RuntimeError(
            f"Expected {horizon} predictions, received {len(forecast)}."
        )
    if quantiles.ndim != 2 or quantiles.shape[0] != horizon or quantiles.shape[1] < 9:
        raise RuntimeError(
            f"Unexpected quantile output shape: {quantiles.shape}."
        )

    # TimesFM returns 9 deciles (0.1 through 0.9).
    return forecast, quantiles[:, 0], quantiles[:, 8]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    symbol = get_coin_pair()
    horizon, requested_unit, actual_minutes = get_forecast_horizon()

    temporary_csv = Path(f"{symbol}_merged_temp.csv")
    forecast_csv = Path(f"{symbol}_forecast.csv")
    forecast_saved = False

    try:
        create_temporary_dataset(symbol, temporary_csv)
        df = load_dataset(temporary_csv)
        model = load_model()

        forecast, p10, p90 = predict_price(df, model, horizon)

        last_time = df["time"].iloc[-1]
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(minutes=BAR_MINUTES),
            periods=horizon,
            freq=f"{BAR_MINUTES}min",
        )

        output = pd.DataFrame({
            "time_utc": future_times,
            "predicted_price": forecast,
            "p10": p10,
            "p90": p90,
        })
        output["time_ist"] = output["time_utc"].dt.tz_convert("Asia/Kolkata")
        output = output[["time_ist", "time_utc", "predicted_price", "p10", "p90"]]
        output.to_csv(forecast_csv, index=False)
        forecast_saved = True

        current_price = float(df["price"].iloc[-1])
        final_price = float(forecast[-1])
        change_pct = ((final_price - current_price) / current_price) * 100.0

        print("\n========== FORECAST RESULT ==========")
        print(f"Pair: {symbol}")
        print(f"Last observed price: {current_price:.8f}")
        print(f"Requested horizon: {requested_unit}")
        print(f"Forecast resolution: {BAR_INTERVAL}")
        print(f"Actual rounded horizon: {actual_minutes} minutes")
        print(f"Final predicted price: {final_price:.8f}")
        print(f"Estimated change: {change_pct:+.3f}%")
        print(f"Forecast CSV saved: {forecast_csv.resolve()}")
        print("\nFirst forecast rows:")
        print(output.head(10).to_string(index=False))
        print("=====================================\n")

    finally:
        # Delete only the intermediate merged input CSV after a successful
        # forecast export. If forecasting fails, keep it for troubleshooting.
        if forecast_saved and DELETE_TEMP_CSV_AFTER_SUCCESS and temporary_csv.exists():
            try:
                temporary_csv.unlink()
                print(f"Deleted temporary input CSV: {temporary_csv}")
            except OSError as exc:
                print(f"Warning: could not delete {temporary_csv}: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as error:
        print(f"\nERROR: {error}")
        print("The temporary merged CSV (if created) is kept when forecasting fails.")




#  python predict.py