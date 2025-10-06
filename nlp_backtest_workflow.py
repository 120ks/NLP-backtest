import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
import numpy as np
from strategy_parser import StrategyParser
from portfolio_backtester import PortfolioBacktester


# === Utility Functions ===

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if returns.std() == 0:
        return 0
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(equity_series):
    cumulative_max = equity_series.cummax()
    drawdown = (equity_series - cumulative_max) / cumulative_max
    return drawdown.min()


def generate_plotly_equity_curve(equity_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode='lines',
        name='Portfolio Equity',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title="📈 Interactive Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_white",
        hovermode="x unified",
        width=900,
        height=450
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def assign_allocation_groups(trades_df, allocation):
    """
    Assign numeric group IDs for overlapping trades that reach full allocation.
    Highlight the final (full allocation) trade with a blue background.
    """
    trades_df = trades_df.sort_values(by="entry_date").reset_index(drop=True)
    trades_df["group_id"] = None
    trades_df["highlight"] = False

    group_id = 1

    for i, trade in trades_df.iterrows():
        overlapping = trades_df[
            (trades_df["entry_date"] < trade["exit_date"]) &
            (trades_df["exit_date"] > trade["entry_date"])
        ]
        active_trades = len(overlapping)

        # Check if this overlap reaches full allocation
        if active_trades * allocation >= 1.0:
            trades_df.loc[overlapping.index, "group_id"] = group_id
            # The current trade that completed full allocation gets a highlight
            trades_df.at[i, "highlight"] = True
            group_id += 1

    return trades_df


# === Main Workflow ===

def main():
    print("\n📈 NLP-Driven Backtesting Workflow (Grouped Allocation Report)")
    print("----------------------------------------------------------------")

    csv_path = "master_stock_data.csv"
    backtester = PortfolioBacktester(csv_path)

    user_input = input(
        "\nEnter strategy (e.g. 'using all tickers in technology sector, whenever one of the tickers drops from its 5 day high by 5%, buy with 50% capital stoploss 5% tp 10%'): "
    )

    parser = StrategyParser()
    strategy = parser.parse(user_input)

    print("\n🧠 Parsed Strategy:")
    print(strategy)

    sector = strategy.get("sector", None)
    entry = strategy.get("entry", {})
    drop_pct = entry.get("condition", {}).get("threshold", 0.05)
    lookback_days = entry.get("condition", {}).get("period_days", 5)
    stop_loss = strategy.get("stop_loss", 0.05)
    take_profit = strategy.get("take_profit", 0.05)
    allocation = strategy.get("allocation", 0.5)

    if not sector:
        print("\n❌ No sector found in input. Please specify one (e.g. 'technology sector').")
        return

    print(f"\n💼 Using allocation={allocation*100:.2f}%, stop_loss={stop_loss*100:.2f}%, take_profit={take_profit*100:.2f}%")

    results = backtester.run_sector_strategy(
        sector_name=sector,
        drop_pct=drop_pct,
        lookback_days=lookback_days,
        allocation_per_trade=allocation,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    trades_df = results.get("trades", pd.DataFrame())
    equity_df = results.get("equity", pd.DataFrame())
    total_return = results.get("total_return", 0.0)
    final_equity = results.get("final_equity", 0.0)

    if trades_df.empty or equity_df.empty:
        print("\n⚠️ No trades generated, report not created.")
        return

    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    trades_df = assign_allocation_groups(trades_df, allocation)

    total_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean()
    avg_trade = trades_df["return"].mean()
    exit_counts = trades_df["exit_reason"].value_counts()

    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)
    sharpe_ratio = calculate_sharpe_ratio(equity_df["returns"])
    max_drawdown = calculate_max_drawdown(equity_df["equity"])

    trades_path = f"reports/trades_{timestamp}.csv"
    trades_df.to_csv(trades_path, index=False)

    # === Build HTML Table ===
    trades_html = """
    <table>
        <tr>
            <th>Group</th><th>Symbol</th><th>Entry Date</th><th>Exit Date</th>
            <th>Entry Price</th><th>Exit Price</th><th>Return</th><th>Exit Reason</th>
        </tr>
    """

    for _, row in trades_df.iterrows():
        row_style = ' style="background-color:#e3f2fd;"' if row["highlight"] else ""
        group_id = row["group_id"] if pd.notna(row["group_id"]) else ""
        trades_html += f"""
        <tr{row_style}>
            <td>{group_id}</td>
            <td>{row['symbol']}</td>
            <td>{row['entry_date'].date()}</td>
            <td>{row['exit_date'].date()}</td>
            <td>{row['entry_price']:.2f}</td>
            <td>{row['exit_price']:.2f}</td>
            <td>{row['return']:.4f}</td>
            <td>{row['exit_reason']}</td>
        </tr>
        """
    trades_html += "</table>"

    plot_html = generate_plotly_equity_curve(equity_df)

    html_path = f"reports/backtest_report_{timestamp}.html"
    html_content = f"""
    <html>
    <head>
        <title>Backtest Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #0d6efd; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            ul {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <h1>📘 Backtest Report (Grouped Allocation)</h1>
        <h3>Strategy:</h3><p>{user_input}</p>
        <h3>Performance Summary:</h3>
        <ul>
            <li>Total Trades: <b>{total_trades}</b></li>
            <li>Win Rate: <b>{win_rate:.2%}</b></li>
            <li>Average Trade Return: <b>{avg_trade:.2%}</b></li>
            <li>Total Return: <b>{total_return:.2%}</b></li>
            <li>Final Equity: <b>${final_equity:,.2f}</b></li>
            <li>Sharpe Ratio: <b>{sharpe_ratio:.2f}</b></li>
            <li>Max Drawdown: <b>{max_drawdown:.2%}</b></li>
        </ul>
        <h3>Exit Breakdown:</h3>
        {exit_counts.to_frame().to_html(header=False)}
        <h3>Equity Curve:</h3>{plot_html}
        <h3>All Trades:</h3>
        <p>🟦 = trade that completed full allocation for its group.</p>
        {trades_html}
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Report saved to: {html_path}")
    print(f"💾 All trades saved to: {trades_path}")
    print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")


if __name__ == "__main__":
    main()
