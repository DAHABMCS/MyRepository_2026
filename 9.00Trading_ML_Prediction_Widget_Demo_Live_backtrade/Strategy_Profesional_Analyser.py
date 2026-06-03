#!/usr/bin/env python3
"""
Momentum Strategy Raw Data Analyzer v2.1
Handles string-based signals (BUY/SELL/EXIT)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class MomentumRawDataAnalyzer:
    def __init__(self, csv_file: str, initial_capital: float = 50000):
        self.csv_file = csv_file
        self.initial_capital = initial_capital
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []

    def load_data(self):
        """Load and validate the CSV data"""
        print(f"\n📂 Loading data from: {self.csv_file}")

        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.data)} rows")
            print(f"📊 Columns: {list(self.data.columns)}")

            # Convert timestamp
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

            # Show unique values in signal columns to understand format
            if 'Entry_Signal' in self.data.columns:
                print(f"\n📋 Entry_Signal unique values: {self.data['Entry_Signal'].unique()}")
            if 'Exit_Signal' in self.data.columns:
                print(f"📋 Exit_Signal unique values: {self.data['Exit_Signal'].unique()}")
            if 'Exit_Reason' in self.data.columns:
                print(f"📋 Exit_Reason unique values: {self.data['Exit_Reason'].unique()}")

            # Check for required columns
            required = ['Close', 'Entry_Signal', 'Exit_Signal']
            missing = [col for col in required if col not in self.data.columns]

            if missing:
                print(f"❌ Missing columns: {missing}")
                return False

            print(f"\n📅 Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            print(f"💰 Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def reconstruct_trades(self):
        """Reconstruct all trades from string-based signals"""
        print("\n🔄 Reconstructing trades from signals...")

        trades = []
        current_trade = None
        in_position = False

        for idx, row in self.data.iterrows():
            entry_signal = str(row.get('Entry_Signal', '')).strip().upper() if pd.notna(row.get('Entry_Signal')) else ''
            exit_signal = str(row.get('Exit_Signal', '')).strip().upper() if pd.notna(row.get('Exit_Signal')) else ''
            exit_reason = str(row.get('Exit_Reason', '')).strip() if pd.notna(row.get('Exit_Reason')) else ''

            # Check for entry signal (BUY)
            if 'BUY' in entry_signal and not in_position:
                # New long trade
                current_trade = {
                    'entry_idx': idx,
                    'entry_time': row['timestamp'],
                    'entry_price': row['Close'],
                    'entry_signal': entry_signal,
                    'risk_allocation': row.get('Risk_Allocation_%', 1.0),
                    'confluence_score': row.get('Confluence_Score', 0),
                    'adx_at_entry': row.get('ADX', 0),
                    'rsi_at_entry': row.get('RSI', 50),
                    'volume_ratio_at_entry': row.get('Volume_Ratio', 1.0),
                    'highest_price': row['Close'],
                    'lowest_price': row['Close'],
                    'bars_held': 0,
                    'exit_reason': None,
                    'position_type': 'LONG'
                }
                in_position = True
                print(f"  🟢 Entry at bar {idx}: ${row['Close']:.2f} ({entry_signal})")

            # Update current trade if in position
            elif in_position and current_trade is not None:
                current_trade['bars_held'] += 1
                current_trade['highest_price'] = max(current_trade['highest_price'], row['Close'])
                current_trade['lowest_price'] = min(current_trade['lowest_price'], row['Close'])

                # Check for exit signal
                has_exit_signal = exit_signal != '' and exit_signal != 'NONE' and exit_signal != '0'
                has_exit_reason = exit_reason != '' and exit_reason != 'nan'

                if has_exit_signal or has_exit_reason or idx == len(self.data) - 1:
                    # Trade ended
                    current_trade.update({
                        'exit_idx': idx,
                        'exit_time': row['timestamp'],
                        'exit_price': row['Close'],
                        'exit_signal': exit_signal,
                        'exit_reason': exit_reason if exit_reason else 'END_OF_DATA',
                        'adx_at_exit': row.get('ADX', 0),
                        'rsi_at_exit': row.get('RSI', 50)
                    })

                    # Calculate trade metrics for LONG position
                    current_trade['pnl'] = (current_trade['exit_price'] - current_trade[
                        'entry_price']) * 100  # Assuming 100 shares
                    current_trade['return_pct'] = (current_trade['exit_price'] - current_trade['entry_price']) / \
                                                  current_trade['entry_price'] * 100
                    current_trade['max_favorable'] = (current_trade['highest_price'] - current_trade['entry_price']) / \
                                                     current_trade['entry_price'] * 100
                    current_trade['max_adverse'] = (current_trade['lowest_price'] - current_trade['entry_price']) / \
                                                   current_trade['entry_price'] * 100

                    current_trade['win'] = current_trade['pnl'] > 0

                    trades.append(current_trade)

                    win_marker = "✅" if current_trade['win'] else "❌"
                    print(f"  {win_marker} Exit at bar {idx}: ${row['Close']:.2f} "
                          f"({current_trade['return_pct']:+.2f}%) in {current_trade['bars_held']} bars "
                          f"[{current_trade['exit_reason']}]")

                    current_trade = None
                    in_position = False

        self.trades = trades
        print(f"\n✅ Reconstructed {len(trades)} trades")

        # Show trade summary
        if trades:
            wins = sum(1 for t in trades if t['win'])
            print(f"📊 Wins: {wins}, Losses: {len(trades) - wins}")
            print(f"📈 Win Rate: {wins / len(trades) * 100:.1f}%")

    def calculate_equity_curve(self):
        """Calculate equity curve from trades"""
        print("\n📈 Calculating equity curve...")

        equity = [self.initial_capital]
        peak = self.initial_capital
        timestamps = [self.data['timestamp'].iloc[0]] if len(self.data) > 0 else []

        # Sort trades by exit
        sorted_trades = sorted(self.trades, key=lambda x: x['exit_idx'])

        for trade in sorted_trades:
            new_equity = equity[-1] + trade['pnl']
            equity.append(new_equity)
            timestamps.append(trade['exit_time'])

            # Update peak
            if new_equity > peak:
                peak = new_equity

            # Calculate drawdown
            dd = (peak - new_equity) / peak * 100
            self.drawdown_curve.append(dd)

        self.equity_curve = equity
        final_pnl = equity[-1] - self.initial_capital
        final_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        print(f"✅ Final equity: ${equity[-1]:,.2f}")
        print(f"💰 Total PnL: ${final_pnl:,.2f} ({final_return:.2f}%)")
        print(f"📉 Max drawdown: {max(self.drawdown_curve) if self.drawdown_curve else 0:.2f}%")

    def calculate_daily_returns(self):
        """Calculate daily returns for Sharpe ratio"""
        if len(self.equity_curve) < 2 or not self.trades:
            return []

        # Create daily equity series
        daily_equity = {}
        current_equity = self.initial_capital
        daily_equity[self.trades[0]['entry_time'].date()] = current_equity

        for trade in sorted(self.trades, key=lambda x: x['exit_time']):
            current_equity += trade['pnl']
            trade_date = trade['exit_time'].date()
            daily_equity[trade_date] = current_equity

        # Calculate daily returns
        daily_returns = []
        dates = sorted(daily_equity.keys())

        for i in range(1, len(dates)):
            prev_equity = daily_equity[dates[i - 1]]
            curr_equity = daily_equity[dates[i]]
            days_diff = (dates[i] - dates[i - 1]).days

            if days_diff > 0:
                # Spread return across days
                daily_return = (curr_equity / prev_equity) ** (1 / days_diff) - 1
                for _ in range(days_diff):
                    daily_returns.append(daily_return)
            else:
                # Same day
                daily_return = (curr_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)

        return daily_returns

    def calculate_metrics(self):
        """Calculate all professional metrics"""
        print("\n" + "=" * 60)
        print("📊 PROFESSIONAL METRICS CALCULATION")
        print("=" * 60)

        metrics = {}

        if not self.trades:
            print("❌ No trades to analyze")
            return metrics

        # 1. Trade statistics
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t['win'])
        losses = total_trades - wins

        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        # 2. P&L statistics
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['win']]) if wins > 0 else 0
        avg_loss = abs(np.mean([t['pnl'] for t in self.trades if not t['win']])) if losses > 0 else 0

        gross_profit = sum(t['pnl'] for t in self.trades if t['win'])
        gross_loss = abs(sum(t['pnl'] for t in self.trades if not t['win']))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 3. Time-based metrics
        if total_trades > 0:
            first_date = min(t['entry_time'] for t in self.trades)
            last_date = max(t['exit_time'] for t in self.trades)
            days_trading = (last_date - first_date).days
            years_trading = days_trading / 365 if days_trading > 0 else 1

            trades_per_year = total_trades / years_trading if years_trading > 0 else total_trades
            total_return_pct = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
            annualized_return = ((1 + total_return_pct / 100) ** (
                        1 / years_trading) - 1) * 100 if years_trading > 0 else 0
        else:
            trades_per_year = 0
            total_return_pct = 0
            annualized_return = 0
            years_trading = 0

        # 4. Max drawdown
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0

        # 5. Sharpe ratio
        daily_returns = self.calculate_daily_returns()
        if len(daily_returns) > 5:  # Need at least some data
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)

            if std_daily_return > 0:
                # Annualized Sharpe (252 trading days)
                sharpe = (avg_daily_return * 252 - 0.02) / (std_daily_return * np.sqrt(252))
            else:
                sharpe = float('inf')
        else:
            sharpe = None

        # 6. Exit reason analysis
        exit_reasons = {}
        for t in self.trades:
            reason = t.get('exit_reason', 'unknown')
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'wins': 0, 'pnl': 0, 'returns': []}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t['pnl']
            exit_reasons[reason]['returns'].append(t['return_pct'])
            if t['win']:
                exit_reasons[reason]['wins'] += 1

        for reason in exit_reasons:
            if exit_reasons[reason]['count'] > 0:
                exit_reasons[reason]['win_rate'] = exit_reasons[reason]['wins'] / exit_reasons[reason]['count'] * 100
                exit_reasons[reason]['avg_return'] = np.mean(exit_reasons[reason]['returns'])

        # 7. Consecutive wins/losses
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        current_type = None

        for t in sorted(self.trades, key=lambda x: x['exit_idx']):
            if t['win']:
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                max_loss_streak = max(max_loss_streak, current_streak)

        # Store metrics
        metrics.update({
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'trades_per_year': trades_per_year,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'years_trading': years_trading,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'exit_reasons': exit_reasons
        })

        return metrics

    def print_results(self, metrics):
        """Print formatted results"""
        if not metrics:
            return

        print("\n" + "=" * 60)
        print("🎯 PROFESSIONAL METRICS RESULTS")
        print("=" * 60)

        # Trade count
        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Trading Period: {metrics['years_trading']:.2f} years")
        print(f"  Trades/Year: {metrics['trades_per_year']:.1f} (Target: 100-120)")

        if metrics['trades_per_year'] >= 100 and metrics['trades_per_year'] <= 120:
            print(f"  ✅ TARGET MET: {metrics['trades_per_year']:.1f} trades/year")
        elif metrics['trades_per_year'] > 120:
            print(f"  ⚠️ ABOVE TARGET: {metrics['trades_per_year']:.1f} trades/year (consider adding filters)")
        else:
            print(f"  ❌ BELOW TARGET: {metrics['trades_per_year']:.1f} trades/year")

        # Win rate
        print(f"\n🎯 WIN RATE:")
        print(f"  Wins: {metrics['wins']}, Losses: {metrics['losses']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")

        if metrics['win_rate'] >= 48 and metrics['win_rate'] <= 55:
            print(f"  ✅ TARGET MET: {metrics['win_rate']:.1f}%")
        elif metrics['win_rate'] > 55:
            print(f"  ⚠️ ABOVE TARGET: {metrics['win_rate']:.1f}% (may be too conservative)")
        else:
            print(f"  ❌ BELOW TARGET: {metrics['win_rate']:.1f}%")

        # P&L metrics
        print(f"\n💰 PROFITABILITY:")
        print(f"  Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"  Avg Win: ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"  Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f} (Target: >1.5)")

        if metrics['profit_factor'] > 1.5:
            print(f"  ✅ TARGET MET: {metrics['profit_factor']:.2f}")
        else:
            print(f"  ❌ BELOW TARGET: {metrics['profit_factor']:.2f}")

        # Risk metrics
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}% (Target: <20%)")

        if metrics['max_drawdown'] < 20:
            print(f"  ✅ TARGET MET: {metrics['max_drawdown']:.2f}%")
        else:
            print(f"  ❌ ABOVE TARGET: {metrics['max_drawdown']:.2f}%")

        if metrics['sharpe_ratio']:
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (Target: >1.5)")
            if metrics['sharpe_ratio'] > 1.5:
                print(f"  ✅ TARGET MET: {metrics['sharpe_ratio']:.2f}")
            else:
                print(f"  ❌ BELOW TARGET: {metrics['sharpe_ratio']:.2f}")
        else:
            print(f"  Sharpe Ratio: N/A (need more data)")

        print(f"\n📈 TRADE QUALITY:")
        print(f"  Max Win Streak: {metrics['max_win_streak']}")
        print(f"  Max Loss Streak: {metrics['max_loss_streak']}")

        # Exit reason analysis
        if metrics['exit_reasons']:
            print(f"\n🚪 EXIT REASON ANALYSIS:")
            # Sort by PnL
            sorted_reasons = sorted(metrics['exit_reasons'].items(),
                                    key=lambda x: x[1]['pnl'], reverse=True)

            for reason, stats in sorted_reasons:
                print(f"\n  {reason}:")
                print(f"    Trades: {stats['count']}")
                print(f"    Win Rate: {stats.get('win_rate', 0):.1f}%")
                print(f"    Avg Return: {stats.get('avg_return', 0):+.2f}%")
                print(f"    Total PnL: ${stats['pnl']:,.2f}")

    def plot_results(self):
        """Plot equity curve and drawdown"""
        if len(self.equity_curve) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        ax1.plot(self.equity_curve, 'b-', linewidth=2, label='Equity')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark trades
        if self.trades:
            # Get equity values at trade exits
            for i, trade in enumerate(sorted(self.trades, key=lambda x: x['exit_idx'])):
                if i < len(self.equity_curve) - 1:
                    color = 'green' if trade['win'] else 'red'
                    ax1.scatter(i + 1, self.equity_curve[i + 1], color=color, s=50,
                                alpha=0.6, zorder=5, edgecolors='black', linewidth=1)

        # Drawdown
        ax2.fill_between(range(len(self.drawdown_curve)), 0,
                         [-d for d in self.drawdown_curve],
                         color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Trade Number')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('momentum_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n📊 Chart saved to: momentum_analysis.png")

    def export_results(self, metrics):
        """Export results to Excel"""
        if not metrics:
            return

        # Create summary DataFrame
        summary_data = [
            {'Metric': 'Total Trades', 'Value': metrics['total_trades']},
            {'Metric': 'Wins', 'Value': metrics['wins']},
            {'Metric': 'Losses', 'Value': metrics['losses']},
            {'Metric': 'Win Rate (%)', 'Value': f"{metrics['win_rate']:.1f}"},
            {'Metric': 'Trades per Year', 'Value': f"{metrics['trades_per_year']:.1f}"},
            {'Metric': 'Total PnL ($)', 'Value': f"${metrics['total_pnl']:,.2f}"},
            {'Metric': 'Total Return (%)', 'Value': f"{metrics['total_return_pct']:.2f}"},
            {'Metric': 'Annualized Return (%)', 'Value': f"{metrics['annualized_return']:.2f}"},
            {'Metric': 'Avg Win ($)', 'Value': f"${metrics['avg_win']:.2f}"},
            {'Metric': 'Avg Loss ($)', 'Value': f"${metrics['avg_loss']:.2f}"},
            {'Metric': 'Win/Loss Ratio', 'Value': f"{metrics['win_loss_ratio']:.2f}"},
            {'Metric': 'Profit Factor', 'Value': f"{metrics['profit_factor']:.2f}"},
            {'Metric': 'Max Drawdown (%)', 'Value': f"{metrics['max_drawdown']:.2f}"},
            {'Metric': 'Sharpe Ratio', 'Value': f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else 'N/A'},
            {'Metric': 'Max Win Streak', 'Value': metrics['max_win_streak']},
            {'Metric': 'Max Loss Streak', 'Value': metrics['max_loss_streak']},
            {'Metric': 'Years Trading', 'Value': f"{metrics['years_trading']:.2f}"}
        ]

        summary_df = pd.DataFrame(summary_data)

        # Create trades DataFrame
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            # Select and reorder columns for better readability
            trade_cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price',
                          'return_pct', 'pnl', 'win', 'bars_held', 'exit_reason',
                          'adx_at_entry', 'rsi_at_entry', 'confluence_score']
            available_cols = [col for col in trade_cols if col in trades_df.columns]
            if available_cols:
                trades_df = trades_df[available_cols]
        else:
            trades_df = pd.DataFrame()

        # Create exit reasons DataFrame
        if metrics['exit_reasons']:
            exit_data = []
            for reason, stats in metrics['exit_reasons'].items():
                exit_data.append({
                    'Exit_Reason': reason,
                    'Trades': stats['count'],
                    'Wins': stats['wins'],
                    'Win_Rate_%': f"{stats.get('win_rate', 0):.1f}",
                    'Total_PnL': f"${stats['pnl']:,.2f}",
                    'Avg_Return_%': f"{stats.get('avg_return', 0):+.2f}"
                })
            exit_df = pd.DataFrame(exit_data)
            exit_df = exit_df.sort_values('Total_PnL', ascending=False)
        else:
            exit_df = pd.DataFrame()

        # Save to Excel
        with pd.ExcelWriter('momentum_detailed_results.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if not trades_df.empty:
                trades_df.to_excel(writer, sheet_name='All_Trades', index=False)
            if not exit_df.empty:
                exit_df.to_excel(writer, sheet_name='Exit_Reasons', index=False)

        print("\n💾 Results saved to: momentum_detailed_results.xlsx")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.load_data():
            return

        self.reconstruct_trades()

        if not self.trades:
            print("❌ No trades found in data")
            return

        self.calculate_equity_curve()
        metrics = self.calculate_metrics()
        self.print_results(metrics)
        self.plot_results()
        self.export_results(metrics)

        # Target achievement summary
        print("\n" + "=" * 60)
        print("🎯 PROFESSIONAL OBJECTIVE ACHIEVEMENT")
        print("=" * 60)

        targets_met = 0
        targets_total = 3  # Trades/year, Sharpe, Max DD

        # Trades/year
        if 100 <= metrics['trades_per_year'] <= 120:
            targets_met += 1
            print(f"✅ Trades/Year: {metrics['trades_per_year']:.1f} (MET)")
        else:
            print(f"❌ Trades/Year: {metrics['trades_per_year']:.1f} (Target: 100-120)")

        # Max Drawdown
        if metrics['max_drawdown'] < 20:
            targets_met += 1
            print(f"✅ Max Drawdown: {metrics['max_drawdown']:.2f}% (MET)")
        else:
            print(f"❌ Max Drawdown: {metrics['max_drawdown']:.2f}% (Target: <20%)")

        # Sharpe Ratio
        if metrics['sharpe_ratio'] and metrics['sharpe_ratio'] > 1.5:
            targets_met += 1
            print(f"✅ Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (MET)")
        else:
            sharpe_str = f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "N/A"
            print(f"❌ Sharpe Ratio: {sharpe_str} (Target: >1.5)")

        print(f"\n📊 Targets Met: {targets_met}/{targets_total}")

        if targets_met == targets_total:
            print("\n🎉 ALL PROFESSIONAL TARGETS ACHIEVED! 🎉")
        else:
            print("\n📝 See recommendations above for improvements")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    print("""
╔════════════════════════════════════════════════════════════════╗
║     MOMENTUM STRATEGY RAW DATA ANALYZER v2.1                   ║
║     Handles string-based signals (BUY/SELL)                    ║
║     Calculates all professional metrics                        ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Get CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("📁 Enter path to your CSV file: ").strip()
        if not csv_file:
            csv_file = "data.csv"  # Default

    # Run analysis
    analyzer = MomentumRawDataAnalyzer(csv_file, initial_capital=50000)
    analyzer.run_analysis()