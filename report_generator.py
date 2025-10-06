import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ReportGenerator:
    """Summarizes and visualizes backtest results, including trade ledger metrics."""

    def __init__(self, df: pd.DataFrame, results: dict):
        self.df = df
        self.results = results
        self.trades = results.get("trades", pd.DataFrame())

    def _extra_metrics_from_trades(self):
        if self.trades is None or self.trades.empty:
            return {
                "Win Rate": 0.0,
                "Avg Win": 0.0,
                "Avg Loss": 0.0,
                "Profit Factor": 0.0,
                "Expectancy": 0.0,
                "Median Hold (days)": 0.0
            }
        r = self.trades["return"]
        wins = r[r > 0]
        losses = r[r <= 0]
        win_rate = (len(wins) / len(r)) if len(r) else 0.0
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0  # negative or zero
        gross_profit = wins.sum() if len(wins) else 0.0
        gross_loss = -losses.sum() if len(losses) else 0.0  # make positive
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        median_hold = self.trades["holding_days"].median() if "holding_days" in self.trades.columns else 0.0

        return {
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Expectancy": expectancy,
            "Median Hold (days)": median_hold
        }

    def summary(self):
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"Symbol: {self.results.get('symbol', 'N/A')}")
        print(f"Action: {self.results.get('action', 'N/A')}")
        print(f"Total Return: {self.results.get('total_return', 0):.2%}")
        print(f"Trades: {self.results.get('num_trades', 0)}")
        print(f"Avg Entry Move: {self.results.get('avg_entry_move', 0):.2%}")
        print(f"Avg Exit  Move: {self.results.get('avg_exit_move', 0):.2%}")

        # Risk stats
        df = self.df.copy()
        df["equity"] = (1 + df["return"].fillna(0)).cumprod()
        max_equity = df["equity"].cummax()
        drawdown = (df["equity"] - max_equity) / max_equity
        max_dd = drawdown.min()
        sharpe = (df["return"].mean() / df["return"].std() * (252 ** 0.5)) if df["return"].std() != 0 else 0.0
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")

        # Trade metrics
        extras = self._extra_metrics_from_trades()
        print(f"Win Rate: {extras['Win Rate']:.2%}")
        print(f"Avg Win: {extras['Avg Win']:.2%}  |  Avg Loss: {extras['Avg Loss']:.2%}")
        pf_str = "∞" if extras["Profit Factor"] == float("inf") else f"{extras['Profit Factor']:.2f}"
        print(f"Profit Factor: {pf_str}")
        print(f"Expectancy (per trade): {extras['Expectancy']:.2%}")
        print(f"Median Hold (days): {extras['Median Hold (days)']:.1f}")
        print("-" * 50)

        # Show a few trades
        if not self.trades.empty:
            print("\n🧾 Sample trades:")
            print(self.trades.head(10).to_string(index=False))

    def plot(self):
        df = self.df.copy()
        if "date" not in df.columns:
            print("⚠️ No date column found — cannot plot.")
            return

        # Price + signals
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(df["date"], df["close"], label="Price", alpha=0.8)
        if "entry_signal" in df.columns:
            ax1.scatter(df.loc[df["entry_signal"], "date"], df.loc[df["entry_signal"], "close"],
                        marker="^", label="Entry", alpha=0.9)
        if "exit_signal" in df.columns:
            ax1.scatter(df.loc[df["exit_signal"], "date"], df.loc[df["exit_signal"], "close"],
                        marker="v", label="Exit", alpha=0.9)
        ax1.set_xlabel("Date"); ax1.set_ylabel("Price"); ax1.legend(loc="upper left")

        # Equity
        ax2 = ax1.twinx()
        if "equity" in df.columns:
            ax2.plot(df["date"], df["equity"], linestyle="--", alpha=0.6, label="Equity")
            ax2.set_ylabel("Equity"); ax2.legend(loc="upper right")

        plt.title(f"{self.results.get('symbol','')} Strategy Performance")
        plt.tight_layout()
        plt.show()
