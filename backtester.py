import pandas as pd
import numpy as np
from typing import Dict, Any, List


class Backtester:
    """
    Backtests entry + exit rules with optional stop-loss/take-profit.
    Produces a trade ledger with entry/exit data.
    Expects CSV columns (your schema is auto-mapped):
      ['date','ticker','country','sector','industry','openprice','closeprice','quartersdividend']
    """

    REQUIRED_COLUMNS = {"date", "symbol", "close"}

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_and_validate_data(csv_path)

    # ---------- data load ----------
    def _load_and_validate_data(self, csv_path: str) -> pd.DataFrame:
        print(f"📂 Loading data from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")
        except Exception as e:
            raise ValueError(f"❌ Error reading CSV: {e}")

        # Normalize + rename
        df.columns = [c.lower().strip() for c in df.columns]
        df.rename(columns={"ticker": "symbol", "closeprice": "close", "openprice": "open"}, inplace=True)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"❌ Missing required columns in CSV even after renaming: {missing}")

        # Dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = int(df["date"].isna().sum())
            print(f"⚠️ Warning: {bad} rows have invalid dates and were dropped.")
            df = df.dropna(subset=["date"])

        df = df.drop_duplicates(subset=["symbol", "date"])
        df.sort_values(["symbol", "date"], inplace=True)
        print(f"✅ Data loaded successfully: {len(df):,} rows, {df['symbol'].nunique()} symbols.\n")
        return df

    # ---------- helpers ----------
    def available_symbols(self):
        if not hasattr(self, "_symbols_cache"):
            self._symbols_cache = sorted(self.data["symbol"].unique().tolist())
        return self._symbols_cache

    # ---------- core backtest ----------
    def run(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        strategy:
          {"entry": {...}, "exit": {...}}  or just a single entry rule
        Returns dict with 'df' (with signals/position) and 'trades' (ledger).
        """
        if "entry" in strategy:
            entry_rule = strategy.get("entry")
            exit_rule = strategy.get("exit")
        else:
            entry_rule = strategy
            exit_rule = None

        symbol = (entry_rule.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("❌ No symbol specified for backtest.")

        df = self.data[self.data["symbol"].str.upper() == symbol].copy()
        if df.empty:
            raise ValueError(f"❌ No data found for symbol '{symbol}'.")

        # ENTRY
        econd = entry_rule["condition"]
        e_period = int((econd.get("period_days") or 3))
        e_thr = econd.get("threshold", 0.05)
        e_dir = econd.get("direction", "down")

        df["entry_change"] = df["close"].pct_change(periods=e_period)
        if e_dir == "down":
            df["entry_signal"] = df["entry_change"] <= -e_thr
        elif e_dir == "up":
            df["entry_signal"] = df["entry_change"] >= e_thr
        else:
            df["entry_signal"] = False

        # EXIT (with stop/take)
        if exit_rule:
            xcond = exit_rule["condition"]
            x_period = int((xcond.get("period_days") or 3))
            x_thr = xcond.get("threshold", 0.03)
            x_dir = xcond.get("direction", "up")
            stop_loss = xcond.get("stop_loss", None)
            take_profit = xcond.get("take_profit", None)

            df["exit_change"] = df["close"].pct_change(periods=x_period)

            # Base exit rule
            if x_dir == "down":
                base_exit = df["exit_change"] <= -x_thr
            elif x_dir == "up":
                base_exit = df["exit_change"] >= x_thr
            else:
                base_exit = pd.Series(False, index=df.index)

            sl_exit = (df["exit_change"] <= -stop_loss) if (stop_loss is not None) else pd.Series(False, index=df.index)
            tp_exit = (df["exit_change"] >= take_profit) if (take_profit is not None) else pd.Series(False, index=df.index)

            df["exit_signal"] = base_exit | sl_exit | tp_exit
            df["exit_reason_base"] = base_exit
            df["exit_reason_sl"] = sl_exit
            df["exit_reason_tp"] = tp_exit
        else:
            df["exit_signal"] = False
            df["exit_reason_base"] = False
            df["exit_reason_sl"] = False
            df["exit_reason_tp"] = False

        # ====== Simulate trades & build ledger ======
        position = 0
        positions: List[int] = []
        ledger = []  # each item: dict with entry/exit details
        entry_idx = None
        entry_price = None
        entry_date = None
        side = 0  # +1 for long, -1 for short

        for i in range(len(df)):
            price = df.iloc[i]["close"]
            date = df.iloc[i]["date"]

            if position == 0 and df.iloc[i]["entry_signal"]:
                # open
                side = 1 if entry_rule["action"] == "buy" else -1
                position = side
                entry_idx = i
                entry_price = price
                entry_date = date

            elif position != 0:
                # check exit
                if df.iloc[i]["exit_signal"]:
                    reason = "rule_exit"
                    if bool(df.iloc[i].get("exit_reason_tp", False)):
                        reason = "take_profit"
                    elif bool(df.iloc[i].get("exit_reason_sl", False)):
                        reason = "stop_loss"

                    # close
                    exit_price = price
                    exit_date = date
                    gross_ret = (exit_price / entry_price - 1.0) * side
                    hold_days = int((exit_date - entry_date).days)

                    ledger.append({
                        "symbol": symbol,
                        "side": "long" if side == 1 else "short",
                        "entry_date": entry_date,
                        "entry_price": float(entry_price),
                        "exit_date": exit_date,
                        "exit_price": float(exit_price),
                        "return": float(gross_ret),
                        "holding_days": hold_days,
                        "exit_reason": reason
                    })

                    # reset
                    position = 0
                    entry_idx = entry_price = entry_date = None
                    side = 0

            positions.append(position)

        # position series + equity from daily returns
        df["position"] = positions
        df["return"] = df["close"].pct_change() * df["position"].shift(1)
        df["equity"] = (1 + df["return"].fillna(0)).cumprod()

        # Metrics
        trades_df = pd.DataFrame(ledger)
        total_return = (1 + df["return"].fillna(0)).prod() - 1
        num_trades = len(trades_df)
        avg_entry_move = df.loc[df["entry_signal"], "entry_change"].mean() if df["entry_signal"].any() else 0.0

        # Exit move (when rule exists)
        if "exit_change" in df.columns:
            avg_exit_move = df.loc[df["exit_signal"], "exit_change"].mean() if df["exit_signal"].any() else 0.0
        else:
            avg_exit_move = 0.0

        print(f"📊 Backtest complete for {symbol}: {num_trades} trades, Total Return {total_return:.2%}")

        return {
            "symbol": symbol,
            "action": entry_rule["action"],
            "total_return": float(total_return),
            "num_trades": int(num_trades),
            "avg_entry_move": float(avg_entry_move or 0.0),
            "avg_exit_move": float(avg_exit_move or 0.0),
            "df": df,
            "trades": trades_df  # << ledger
        }
