"""
Monte Carlo Simulation Module for Backtesting
Adds probabilistic analysis to backtest results with capacity limits and optional log-scale plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtest validation and risk analysis.
    Generates multiple random walk scenarios based on historical trade statistics
    to estimate probability distributions of future outcomes.
    """

    def __init__(self, trade_history, initial_capital=50000):
        """
        Initialize Monte Carlo simulator.

        Args:
            trade_history: List of completed trades from backtest
            initial_capital: Starting capital amount
        """
        self.trade_history = trade_history
        self.initial_capital = initial_capital
        self.trade_returns = []
        self.trade_stats = {}

        self._calculate_trade_statistics()

    def _calculate_trade_statistics(self):
        """Extract statistical properties from historical trades"""
        if not self.trade_history:
            raise ValueError("No trade history provided")

        skipped = 0
        for trade in self.trade_history:
            if 'pnl_pct' in trade:
                # Already expressed as a % return — use as-is (backward compatible)
                self.trade_returns.append(trade['pnl_pct'] / 100)
            elif 'profit' in trade:
                equity_basis = trade.get('equity_at_entry', self.initial_capital)
                if equity_basis and equity_basis > 0:
                    self.trade_returns.append(trade['profit'] / equity_basis)
                else:
                    skipped += 1
            else:
                skipped += 1

        if skipped:
            print(f"⚠️  Skipped {skipped} trade(s) missing 'profit'/'pnl_pct' data")

        if not self.trade_returns:
            raise ValueError("No completed trades with PnL data found")

        # Calculate distribution parameters
        self.trade_stats = {
            'mean_return': np.mean(self.trade_returns),
            'std_return': np.std(self.trade_returns),
            'win_rate': sum(1 for r in self.trade_returns if r > 0) / len(self.trade_returns),
            'avg_win': np.mean([r for r in self.trade_returns if r > 0]) if any(
                r > 0 for r in self.trade_returns) else 0,
            'avg_loss': np.mean([r for r in self.trade_returns if r <= 0]) if any(
                r <= 0 for r in self.trade_returns) else 0,
            'total_trades': len(self.trade_returns),
            'max_return': max(self.trade_returns),
            'min_return': min(self.trade_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }

        print(f"📊 Trade Statistics Extracted:")
        print(f"   Total Trades: {self.trade_stats['total_trades']}")
        print(f"   Win Rate: {self.trade_stats['win_rate'] * 100:.2f}%")
        print(f"   Avg Return: {self.trade_stats['mean_return'] * 100:.2f}%")
        print(f"   Volatility: {self.trade_stats['std_return'] * 100:.2f}%")

    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from returns"""
        if len(self.trade_returns) < 2:
            return 0

        excess_returns = np.array(self.trade_returns)
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

    def run_simulation(self, n_simulations=1000, n_trades=1000, method='parametric', max_position_capital=None):
        """
        Run Monte Carlo simulation with multiple methods.

        Args:
            n_simulations: Number of random paths to generate
            n_trades: Number of trades per simulation
            method: 'parametric', 'bootstrap', or 'hybrid'
            max_position_capital: Maximum allowed capital per trade position (None for unconstrained geometric compounding)

        Returns:
            dict: Simulation results with equity curves and statistics
        """
        print(f"\n🎲 Running Monte Carlo Simulation...")
        print(f"   Method: {method.upper()}")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Trades per path: {n_trades}")
        if max_position_capital:
            print(f"   Max Position Capital Cap: ${max_position_capital:,.2f}")

        simulation_results = {
            'equity_curves': [],
            'final_capitals': [],
            'max_drawdowns': [],
            'sharpe_ratios': [],
            'win_rates': []
        }

        for i in range(n_simulations):
            equity_curve, trades = self._generate_path(n_trades, method, max_position_capital)

            simulation_results['equity_curves'].append(equity_curve)
            simulation_results['final_capitals'].append(equity_curve[-1])

            max_dd = self._calculate_max_drawdown(equity_curve)
            simulation_results['max_drawdowns'].append(max_dd)

            eq_array = np.array(equity_curve)
            returns = np.diff(eq_array) / eq_array[:-1]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            simulation_results['sharpe_ratios'].append(sharpe)

            wins = sum(1 for t in trades if t > 0)
            simulation_results['win_rates'].append(wins / len(trades) if trades else 0)

            if (i + 1) % max(1, (n_simulations // 10)) == 0:
                print(f"   Progress: {(i + 1) / n_simulations * 100:.0f}%")

        print("✅ Simulation complete!")

        simulation_results['statistics'] = self._calculate_statistics(simulation_results)

        return simulation_results

    def _generate_path(self, n_trades, method='parametric', max_position_capital=None):
        """Generate single simulation path using specified method"""
        equity = [self.initial_capital]
        trades = []

        for _ in range(n_trades):
            if method == 'parametric':
                trade_return = np.random.normal(
                    self.trade_stats['mean_return'],
                    self.trade_stats['std_return']
                )

            elif method == 'bootstrap':
                trade_return = np.random.choice(self.trade_returns)

            elif method == 'hybrid':
                if np.random.random() < 0.5:
                    trade_return = np.random.normal(
                        self.trade_stats['mean_return'],
                        self.trade_stats['std_return']
                    )
                else:
                    trade_return = np.random.choice(self.trade_returns)

            else:
                raise ValueError(f"Unknown method: {method}")

            current_equity = equity[-1]

            # Apply position capacity constraint if provided
            if max_position_capital is not None:
                allocated_capital = min(current_equity, max_position_capital)
                pnl = allocated_capital * trade_return
                new_equity = max(0.0, current_equity + pnl)
            else:
                new_equity = max(0.0, current_equity * (1 + trade_return))

            equity.append(new_equity)
            trades.append(trade_return)

        return equity, trades

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd

        return max_dd * 100

    def _calculate_statistics(self, results):
        """Calculate statistical summary of simulation results"""
        stats_dict = {}

        final_caps = results['final_capitals']
        stats_dict['final_capital'] = {
            'mean': np.mean(final_caps),
            'median': np.median(final_caps),
            'std': np.std(final_caps),
            'min': np.min(final_caps),
            'max': np.max(final_caps),
            'percentile_5': np.percentile(final_caps, 5),
            'percentile_95': np.percentile(final_caps, 95),
            'prob_profit': sum(1 for c in final_caps if c > self.initial_capital) / len(final_caps)
        }

        returns = [(c - self.initial_capital) / self.initial_capital * 100 for c in final_caps]
        stats_dict['returns'] = {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_95': np.percentile(returns, 95)
        }

        stats_dict['drawdown'] = {
            'mean': np.mean(results['max_drawdowns']),
            'median': np.median(results['max_drawdowns']),
            'worst': np.max(results['max_drawdowns']),
            'percentile_95': np.percentile(results['max_drawdowns'], 95)
        }

        stats_dict['sharpe'] = {
            'mean': np.mean(results['sharpe_ratios']),
            'median': np.median(results['sharpe_ratios']),
            'percentile_5': np.percentile(results['sharpe_ratios'], 5)
        }

        return stats_dict

    def visualize_results(self, results, save_path='monte_carlo_analysis.png', log_scale=False):
        """Create comprehensive visualization of Monte Carlo results"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity Curves
        ax1 = fig.add_subplot(gs[0:2, :2])
        self._plot_equity_curves(ax1, results, log_scale=log_scale)

        # 2. Final Capital Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_final_capital_distribution(ax2, results)

        # 3. Drawdown Distribution
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_drawdown_distribution(ax3, results)

        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_return_distribution(ax4, results)

        # 5. Probability Cone
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_probability_cone(ax5, results, log_scale=log_scale)

        # 6. Statistics Summary
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_statistics_summary(ax6, results)

        plt.suptitle('Monte Carlo Simulation Results',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Monte Carlo visualization saved to: {save_path}")
        plt.show()

    def _plot_equity_curves(self, ax, results, log_scale=False):
        """Plot all simulated equity curves with percentiles"""
        curves = np.array(results['equity_curves'])
        n_trades = curves.shape[1]

        sample_size = min(100, len(curves))
        sample_indices = np.random.choice(len(curves), sample_size, replace=False)

        for idx in sample_indices:
            ax.plot(curves[idx], alpha=0.1, color='blue', linewidth=0.5)

        p50 = np.percentile(curves, 50, axis=0)
        p5 = np.percentile(curves, 5, axis=0)
        p95 = np.percentile(curves, 95, axis=0)

        ax.plot(p50, color='red', linewidth=2, label='Median (50th percentile)')
        ax.fill_between(range(n_trades), p5, p95, alpha=0.3, color='red',
                        label='90% Confidence Interval')

        ax.axhline(y=self.initial_capital, color='green', linestyle='--',
                   linewidth=2, label=f'Initial Capital: ${self.initial_capital:,.0f}')

        ax.set_xlabel('Trade Number', fontsize=11)
        ax.set_ylabel('Equity ($)', fontsize=11)
        ax.set_title('Simulated Equity Curves', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='plain', axis='y')

    def _plot_final_capital_distribution(self, ax, results):
        """Plot distribution of final capital values"""
        final_caps = results['final_capitals']
        stats_cap = results['statistics']['final_capital']

        ax.hist(final_caps, bins=50, alpha=0.7, color='blue', edgecolor='black')

        ax.axvline(stats_cap['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: ${stats_cap['mean']:,.0f}")
        ax.axvline(stats_cap['median'], color='green', linestyle='--',
                   linewidth=2, label=f"Median: ${stats_cap['median']:,.0f}")
        ax.axvline(self.initial_capital, color='orange', linestyle='--',
                   linewidth=2, label=f"Initial: ${self.initial_capital:,.0f}")

        ax.set_xlabel('Final Capital ($)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Final Capital Distribution\nProfit Prob: {stats_cap["prob_profit"] * 100:.1f}%',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_drawdown_distribution(self, ax, results):
        """Plot distribution of maximum drawdowns"""
        drawdowns = results['max_drawdowns']
        stats_dd = results['statistics']['drawdown']

        ax.hist(drawdowns, bins=50, alpha=0.7, color='red', edgecolor='black')

        ax.axvline(stats_dd['mean'], color='blue', linestyle='--',
                   linewidth=2, label=f"Mean: {stats_dd['mean']:.2f}%")
        ax.axvline(stats_dd['worst'], color='darkred', linestyle='--',
                   linewidth=2, label=f"Worst: {stats_dd['worst']:.2f}%")

        ax.set_xlabel('Max Drawdown (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Maximum Drawdown Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_return_distribution(self, ax, results):
        """Plot distribution of returns with normal curve overlay"""
        final_caps = results['final_capitals']
        returns = [(c - self.initial_capital) / self.initial_capital * 100 for c in final_caps]
        stats_ret = results['statistics']['returns']

        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, color='green',
                                   edgecolor='black', density=True)

        mu, sigma = stats_ret['mean'], stats_ret['std']
        x = np.linspace(min(returns), max(returns), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal(μ={mu:.2f}%, σ={sigma:.2f}%)')

        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Break-even')

        ax.set_xlabel('Return (%)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title('Return Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_probability_cone(self, ax, results, log_scale=False):
        """Plot probability cone showing likely equity paths"""
        curves = np.array(results['equity_curves'])

        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

        for i, p in enumerate(percentiles):
            values = np.percentile(curves, p, axis=0)
            ax.plot(values, color=colors[i], linewidth=2,
                    label=f'{p}th percentile', alpha=0.7)

        ax.axhline(y=self.initial_capital, color='blue', linestyle='--',
                   linewidth=2, label='Initial Capital')

        ax.set_xlabel('Trade Number', fontsize=10)
        ax.set_ylabel('Equity ($)', fontsize=10)
        ax.set_title('Probability Cone', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='plain', axis='y')

    def _plot_statistics_summary(self, ax, results):
        """Display key statistics as text"""
        ax.axis('off')

        stats_cap = results['statistics']['final_capital']
        stats_ret = results['statistics']['returns']
        stats_dd = results['statistics']['drawdown']

        summary_text = f"""
MONTE CARLO SUMMARY
{'=' * 30}

FINAL CAPITAL
  Mean:        ${stats_cap['mean']:,.0f}
  Median:      ${stats_cap['median']:,.0f}
  5th-95th:    ${stats_cap['percentile_5']:,.0f} - ${stats_cap['percentile_95']:,.0f}
  Profit Prob: {stats_cap['prob_profit'] * 100:.1f}%

RETURNS
  Mean:        {stats_ret['mean']:.2f}%
  Median:      {stats_ret['median']:.2f}%
  5th-95th:    {stats_ret['percentile_5']:.2f}% - {stats_ret['percentile_95']:.2f}%

DRAWDOWN
  Mean:        {stats_dd['mean']:.2f}%
  Median:      {stats_dd['median']:.2f}%
  Worst:       {stats_dd['worst']:.2f}%

RISK METRICS
  Sharpe (Avg): {results['statistics']['sharpe']['mean']:.3f}
"""

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def generate_report(self, results, output_path='monte_carlo_report.txt'):
        """Generate detailed text report of simulation results"""
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MONTE CARLO SIMULATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Simulations Run: {len(results['final_capitals']):,}\n")
            f.write(f"Trades per Simulation: {len(results['equity_curves'][0]) - 1}\n\n")

            f.write("=" * 70 + "\n")
            f.write("HISTORICAL TRADE STATISTICS\n")
            f.write("=" * 70 + "\n\n")

            for key, value in self.trade_stats.items():
                if isinstance(value, float):
                    if 'rate' in key or 'return' in key:
                        f.write(f"{key:20} {value * 100:8.2f}%\n")
                    else:
                        f.write(f"{key:20} {value:8.4f}\n")
                else:
                    f.write(f"{key:20} {value}\n")

            f.write("\n")

            stats_cap = results['statistics']['final_capital']
            stats_ret = results['statistics']['returns']
            stats_dd = results['statistics']['drawdown']
            stats_sharpe = results['statistics']['sharpe']

            f.write("=" * 70 + "\n")
            f.write("SIMULATION RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("FINAL CAPITAL:\n")
            f.write(f"  Mean:                    ${stats_cap['mean']:,.2f}\n")
            f.write(f"  Median:                  ${stats_cap['median']:,.2f}\n")
            f.write(f"  Standard Deviation:      ${stats_cap['std']:,.2f}\n")
            f.write(f"  Range:                   ${stats_cap['min']:,.2f} - ${stats_cap['max']:,.2f}\n")
            f.write(
                f"  90% Confidence Interval: ${stats_cap['percentile_5']:,.2f} - ${stats_cap['percentile_95']:,.2f}\n")
            f.write(f"  Probability of Profit:   {stats_cap['prob_profit'] * 100:.2f}%\n\n")

            f.write("RETURNS:\n")
            f.write(f"  Mean:                    {stats_ret['mean']:.2f}%\n")
            f.write(f"  Median:                  {stats_ret['median']:.2f}%\n")
            f.write(f"  Standard Deviation:      {stats_ret['std']:.2f}%\n")
            f.write(
                f"  90% Confidence Interval: {stats_ret['percentile_5']:.2f}% - {stats_ret['percentile_95']:.2f}%\n\n")

            f.write("MAXIMUM DRAWDOWN:\n")
            f.write(f"  Mean:                    {stats_dd['mean']:.2f}%\n")
            f.write(f"  Median:                  {stats_dd['median']:.2f}%\n")
            f.write(f"  Worst Case (95th %ile):  {stats_dd['percentile_95']:.2f}%\n")
            f.write(f"  Absolute Worst:          {stats_dd['worst']:.2f}%\n\n")

            f.write("SHARPE RATIO:\n")
            f.write(f"  Mean:                    {stats_sharpe['mean']:.3f}\n")
            f.write(f"  Median:                  {stats_sharpe['median']:.3f}\n")
            f.write(f"  5th Percentile:          {stats_sharpe['percentile_5']:.3f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("RISK ASSESSMENT\n")
            f.write("=" * 70 + "\n\n")

            risk_of_ruin = sum(1 for c in results['final_capitals'] if c < self.initial_capital * 0.5) / len(
                results['final_capitals'])
            f.write(f"Risk of 50% Loss:        {risk_of_ruin * 100:.2f}%\n")

            breakeven_prob = sum(1 for c in results['final_capitals'] if c >= self.initial_capital) / len(
                results['final_capitals'])
            f.write(f"Probability of Profit:   {breakeven_prob * 100:.2f}%\n")

            double_prob = sum(1 for c in results['final_capitals'] if c >= self.initial_capital * 2) / len(
                results['final_capitals'])
            f.write(f"Probability of 2x:       {double_prob * 100:.2f}%\n\n")

        print(f"✅ Monte Carlo report saved to: {output_path}")