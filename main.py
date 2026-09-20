import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Fetch data and clean MultiIndex column structures if present
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Close", level=0, axis=1) if "Close" in df.columns.levels[0] else df.droplevel(1, axis=1)
    else:
        df = df[["Close"]]
    return df.copy()


if __name__ == "__main__":
    # 1. Fetch wider range to give SMA_200 enough warm-up data
    df = get_data("NVDA", start="2021-01-01", end="2026-01-01")

    # If df is a DataFrame with column name 'NVDA', convert/rename to 'Close'
    if isinstance(df, pd.DataFrame):
        df.columns = ["Close"]

    # 2. Indicators
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # 3. Execution Signal (1 = Long, 0 = Cash)
    df["Signal"] = np.where(df["SMA_20"] > df["SMA_200"], 1, 0)
    # Shift position by 1 day to trade on the next close (prevents lookahead bias)
    df["Position"] = df["Signal"].shift(1)

    # 4. Returns
    df["Market_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]

    # Drop warm-up rows where indicators/returns are NaN
    clean_df = df.dropna(subset=["SMA_200", "Strategy_Return"]).copy()

    # Cumulative Returns
    clean_df["Cum_Market_Return"] = (1 + clean_df["Market_Return"]).cumprod()
    clean_df["Cum_Strategy_Return"] = (1 + clean_df["Strategy_Return"]).cumprod()

    # 5. Risk / Drawdown Logic
    clean_df["Peak"] = clean_df["Cum_Strategy_Return"].cummax()
    clean_df["Drawdown"] = (clean_df["Cum_Strategy_Return"] - clean_df["Peak"]) / clean_df["Peak"]

    max_drawdown = clean_df["Drawdown"].min()
    total_strategy_return = (clean_df["Cum_Strategy_Return"].iloc[-1] - 1) * 100
    total_market_return = (clean_df["Cum_Market_Return"].iloc[-1] - 1) * 100

    # Output Summary
    print(f"Strategy Total Return: {total_strategy_return:.2f}%")
    print(f"Buy & Hold Return:     {total_market_return:.2f}%")
    print(f"Maximum Drawdown:       {max_drawdown * 100:.2f}%")

    # 6. Visualization with Matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Price & Moving Averages
    ax1.plot(clean_df.index, clean_df["Close"], label="NVDA Close", color="black", alpha=0.6)
    ax1.plot(clean_df.index, clean_df["SMA_20"], label="20-Day SMA", color="blue", linestyle="--")
    ax1.plot(clean_df.index, clean_df["SMA_200"], label="200-Day SMA", color="orange", linestyle="--")
    ax1.set_title("NVDA Price & Moving Average Crossover Strategy")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative Returns Comparison
    ax2.plot(clean_df.index, clean_df["Cum_Strategy_Return"], label="SMA Crossover Strategy", color="green",
             linewidth=1.5)
    ax2.plot(clean_df.index, clean_df["Cum_Market_Return"], label="Buy & Hold (NVDA)", color="gray", linestyle=":",
             linewidth=1.5)
    ax2.set_title("Cumulative Growth of $1 Invested")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Growth Factor")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()