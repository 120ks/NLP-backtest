import pandas as pd
from typing import Dict, Any, List


class PortfolioBacktester:
    """
    Sector-aware backtester supporting multi-ticker portfolio simulation.
    Buys when today's open <= (1 - drop_pct) * trailing N-day high (previous days only).
    Exits on take-profit or stop-loss.
    """

    def __init__(self, csv_path: str, starting_capital: float = 100_000.0):
        self.csv_path = csv_path
        self.starting_capital = starting_capital
        self.data = self._load_and_prepare_data(csv_path)

    # -------------------------------------------------------------
    def _load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        rename_map = {
            "ticker": "symbol",
            "closeprice": "close",
            "openprice": "open"
        }
        df.rename(columns=rename_map, inplace=True)

        required_cols = {"date", "symbol", "close", "open"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "symbol", "open", "close"], inplace=True)
        df.sort_values(["symbol", "date"], inplace=True)

        if "sector" in df.columns:
            df["sector"] = df["sector"].astype(str).str.lower()
        else:
            df["sector"] = "unknown"

        return df

    # -------------------------------------------------------------
    def run_sector_strategy(self,
                            sector_name: str,
                            drop_pct: float = 0.05,
                            lookback_days: int = 5,
                            allocation_per_trade: float = 0.5,
                            stop_loss: float = 0.05,
                            take_profit: float = 0.05):
        """
        Runs a backtest where a buy occurs when today's OPEN <= (1 - drop_pct) * trailing N-day high.
        Uses trailing logic (previous N days only, no lookahead).
        """

        df = self.data[self.data["sector"].str.lower() == sector_name.lower()].copy()
        if df.empty:
            raise ValueError(f"No data found for sector '{sector_name}'")

        symbols = sorted(df["symbol"].unique().tolist())
        print(f"Running strategy for sector '{sector_name}' on {len(symbols)} symbols...")

        # ✅ compute trailing high (exclude today's price)
        df["trailing_high"] = df.groupby("symbol")["close"].transform(
            lambda x: x.shift(1).rolling(int(lookback_days), min_periods=1).max()
        )

        # compute drop condition based on open price
        df["drop_from_high"] = (df["open"] / df["trailing_high"]) - 1.0

        cash = float(self.starting_capital)
        equity_curve: List[Dict[str, Any]] = []
        positions: Dict[str, Dict[str, Any]] = {}
        trades: List[Dict[str, Any]] = []

        dates = sorted(df["date"].unique())

        for date in dates:
            day_data = df[df["date"] == date]

            # ---- exit positions ----
            for sym in list(positions.keys()):
                pos = positions[sym]
                sub = day_data[day_data["symbol"] == sym]
                if sub.empty:
                    continue
                price = float(sub.iloc[0]["close"])
                ret = (price / pos["entry_price"]) - 1.0

                if ret >= take_profit:
                    exit_reason = "take_profit"
                elif ret <= -stop_loss:
                    exit_reason = "stop_loss"
                else:
                    continue  # still open

                proceeds = price * pos["shares"]
                cash += proceeds
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "return": ret,
                    "exit_reason": exit_reason
                })
                del positions[sym]

            # ---- entry conditions ----
            open_trades = len(positions)
            if open_trades * allocation_per_trade < 1.0:
                candidates = day_data[
                    (day_data["drop_from_high"] <= -drop_pct) &
                    (~day_data["symbol"].isin(positions.keys()))
                ]
                if not candidates.empty:
                    available_fraction = 1.0 - (open_trades * allocation_per_trade)
                    max_new = int(available_fraction // allocation_per_trade)
                    for _, row in candidates.head(max_new).iterrows():
                        invest_amt = self.starting_capital * allocation_per_trade
                        if cash < invest_amt:
                            continue
                        shares = invest_amt / float(row["open"])
                        cash -= invest_amt
                        positions[row["symbol"]] = {
                            "entry_price": float(row["open"]),
                            "shares": float(shares),
                            "entry_date": date
                        }

            # ---- update equity ----
            mkt_val = 0.0
            for sym, pos in positions.items():
                sub = day_data[day_data["symbol"] == sym]
                if sub.empty:
                    continue
                mkt_val += pos["shares"] * float(sub.iloc[0]["close"])

            equity_curve.append({"date": date, "equity": cash + mkt_val})

        equity_df = pd.DataFrame(equity_curve)
        total_return = equity_df["equity"].iloc[-1] / self.starting_capital - 1.0

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            earliest = trades_df["entry_date"].min().date()
            latest = trades_df["exit_date"].max().date()
            print(f"✅ Finished: {len(trades_df)} trades from {earliest} → {latest}")
        else:
            print("⚠️ No trades triggered. Try smaller drop_pct or longer lookback_days.")

        print(f"Final Equity: ${equity_df['equity'].iloc[-1]:,.2f}  |  Total Return {total_return:.2%}")

        return {
            "equity": equity_df,
            "trades": trades_df,
            "final_equity": float(equity_df['equity'].iloc[-1]),
            "total_return": float(total_return),
            "sector": sector_name,
            "symbols": symbols
        }
