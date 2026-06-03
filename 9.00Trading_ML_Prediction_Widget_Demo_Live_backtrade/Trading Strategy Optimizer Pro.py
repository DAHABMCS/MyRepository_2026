import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import itertools
import json
import os

warnings.filterwarnings('ignore')

# Load your actual data
print("=" * 80)
print("LOADING DATA...")
print("=" * 80)

df = pd.read_csv('2025-2026.csv', header=None,
                 names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"Data loaded successfully!")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Total bars: {len(df):,}")
print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
print(f"Current price: ${df['close'].iloc[-1]:.2f}")

# Calculate overall trend
first_price = df['close'].iloc[0]
last_price = df['close'].iloc[-1]
trend_pct = (last_price - first_price) / first_price * 100
print(f"Overall trend: {trend_pct:.2f}% {'(BEARISH)' if trend_pct < 0 else '(BULLISH)'}")


class AdvancedTradingStrategy:
    """Advanced trading strategy with optimization capabilities"""

    def __init__(self, name, initial_capital=100000, commission=0.001):
        self.name = name
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.current_position = None
        self.equity_curve = []
        self.params = {}

    def reset(self):
        self.capital = self.initial_capital
        self.positions = []
        self.current_position = None
        self.equity_curve = []

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.positions) == 0:
            return {
                'total_return': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
                'max_drawdown': 0, 'win_rate': 0, 'profit_factor': 0,
                'num_trades': 0, 'avg_win': 0, 'avg_loss': 0, 'avg_trade_return': 0,
                'calmar_ratio': 0, 'expectancy': 0, 'recovery_factor': 0
            }

        # Calculate trade statistics
        trade_returns = []
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0
        gross_loss = 0
        trade_durations = []

        for pos in self.positions:
            ret = (pos['exit_price'] - pos['entry_price']) / pos['entry_price']
            if pos['direction'] == 'short':
                ret = -ret
            trade_returns.append(ret)

            duration = (pos['exit_time'] - pos['entry_time']).total_seconds() / 3600
            trade_durations.append(duration)

            pnl_abs = ret * pos['entry_price'] * pos['size']
            if ret > 0:
                winning_trades += 1
                gross_profit += pnl_abs
            else:
                losing_trades += 1
                gross_loss += abs(pnl_abs)

        # Calculate equity curve returns
        equity = pd.Series(self.equity_curve)
        equity_returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio (assuming 252 trading days * 96 15-min bars = 24192 periods/year)
        periods_per_year = 252 * 96
        sharpe = np.sqrt(periods_per_year) * equity_returns.mean() / equity_returns.std() if len(
            equity_returns) > 0 and equity_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = equity_returns[equity_returns < 0]
        sortino = np.sqrt(periods_per_year) * equity_returns.mean() / downside_returns.std() if len(
            downside_returns) > 0 and downside_returns.std() > 0 else 0

        # Calculate max drawdown
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        # Calmar ratio (return / max drawdown)
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Expectancy (average profit per trade)
        expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if avg_loss > 0 else 0

        # Recovery factor (total profit / max drawdown)
        recovery = (self.capital - self.initial_capital) / abs(
            max_drawdown * self.initial_capital) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.positions),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_return': avg_trade_return,
            'calmar_ratio': calmar,
            'expectancy': expectancy,
            'recovery_factor': recovery,
            'avg_duration_hours': np.mean(trade_durations) if trade_durations else 0
        }

    @staticmethod
    def calculate_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_adx(df, period=14):
        high = df['high']
        low = df['low']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)

        tr = AdvancedTradingStrategy.calculate_atr(df, 1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx


class OptimizedShortStrategy(AdvancedTradingStrategy):
    """Specialized strategy for bearish markets - optimized for short trades"""

    def __init__(self, name, fast_ma=8, slow_ma=21, rsi_period=14,
                 atr_period=14, stop_atr=1.5, take_profit_atr=2.5,
                 min_adx=20, risk_per_trade=0.02):
        super().__init__(name)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.take_profit_atr = take_profit_atr
        self.min_adx = min_adx
        self.risk_per_trade = risk_per_trade
        self.params = {
            'fast_ma': fast_ma, 'slow_ma': slow_ma, 'rsi_period': rsi_period,
            'atr_period': atr_period, 'stop_atr': stop_atr,
            'take_profit_atr': take_profit_atr, 'min_adx': min_adx,
            'risk_per_trade': risk_per_trade
        }

    def backtest(self, df):
        self.reset()

        df = df.copy()

        # Calculate indicators
        df['ma_fast'] = df['close'].rolling(self.fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_ma).mean()
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = self.calculate_atr(df, self.atr_period)
        df['adx'] = self.calculate_adx(df, 14)

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(5)

        # Generate short signals
        df['short_signal'] = (
                (df['ma_fast'] < df['ma_slow']) &  # Bearish crossover
                (df['close'] < df['ma_fast']) &  # Price below fast MA
                (df['rsi'] < 70) &  # Not overbought (allows continuation)
                (df['momentum'] < 0) &  # Negative momentum
                (df['adx'] > self.min_adx) &  # Strong trend
                (df['volume_ratio'] > 0.8)  # Decent volume
        ).astype(int)

        # Exit signals
        df['exit_signal'] = (
                (df['ma_fast'] > df['ma_slow']) |  # Bullish crossover
                (df['rsi'] < 30) |  # Oversold
                (df['momentum'] > 0)  # Positive momentum
        ).astype(int)

        # Track trailing high for dynamic stop
        trailing_high = None

        for i in range(max(self.slow_ma, self.rsi_period, self.atr_period, 30), len(df)):
            # Entry logic (short only)
            if df['short_signal'].iloc[i] == 1 and self.current_position is None:
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price + self.stop_atr * df['atr'].iloc[i]
                take_profit = entry_price - self.take_profit_atr * df['atr'].iloc[i]

                # Position sizing based on risk
                position_size = int((self.capital * self.risk_per_trade) / (stop_loss - entry_price))
                position_size = min(position_size, int(self.capital / entry_price))

                if position_size > 0:
                    self.current_position = {
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'direction': 'short',
                        'trailing_high': entry_price,
                        'max_favorable': entry_price
                    }
                    trailing_high = entry_price

            # Exit logic
            elif self.current_position is not None:
                exit_signal = False
                exit_price = None
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]

                # Update trailing high
                trailing_high = max(trailing_high, current_high) if trailing_high else current_high

                # Trailing stop (tightens as trade progresses)
                trailing_stop = trailing_high - 0.8 * df['atr'].iloc[i]

                # Check exits
                if current_low <= self.current_position['take_profit']:
                    exit_price = self.current_position['take_profit']
                    exit_signal = True
                elif current_high >= self.current_position['stop_loss']:
                    exit_price = self.current_position['stop_loss']
                    exit_signal = True
                elif current_high >= trailing_stop:
                    exit_price = trailing_stop
                    exit_signal = True
                elif df['exit_signal'].iloc[i] == 1:
                    exit_price = df['close'].iloc[i]
                    exit_signal = True

                if exit_signal:
                    pnl = (self.current_position['entry_price'] - exit_price) * self.current_position['size']

                    self.capital += pnl * (1 - self.commission)
                    self.positions.append({
                        **self.current_position,
                        'exit_time': df.index[i],
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    self.current_position = None
                    trailing_high = None

            self.equity_curve.append(self.capital)

        return self.calculate_metrics()


class HybridOptimizedStrategy(AdvancedTradingStrategy):
    """Hybrid strategy combining multiple signals with optimization"""

    def __init__(self, name, fast_ma=5, slow_ma=15, ema_short=9, ema_long=21,
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 atr_period=14, stop_atr=1.5, take_profit_atr=2.2,
                 min_adx=20, volume_threshold=1.1, risk_per_trade=0.02):
        super().__init__(name)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.take_profit_atr = take_profit_atr
        self.min_adx = min_adx
        self.volume_threshold = volume_threshold
        self.risk_per_trade = risk_per_trade

    def backtest(self, df):
        self.reset()

        df = df.copy()

        # Technical indicators
        df['ma_fast'] = df['close'].rolling(self.fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_ma).mean()
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = self.calculate_atr(df, self.atr_period)
        df['adx'] = self.calculate_adx(df, 14)

        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['momentum_pct'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100

        # Generate signals (optimized for bearish bias)
        df['short_signal'] = (
                (df['ma_fast'] < df['ma_slow']) &
                (df['ema_short'] < df['ema_long']) &
                (df['macd_hist'] < 0) &
                (df['rsi'] < self.rsi_overbought) &
                (df['adx'] > self.min_adx) &
                (df['volume_ratio'] > self.volume_threshold)
        ).astype(int)

        df['long_signal'] = (
                (df['ma_fast'] > df['ma_slow']) &
                (df['ema_short'] > df['ema_long']) &
                (df['macd_hist'] > 0) &
                (df['rsi'] > self.rsi_oversold) &
                (df['adx'] > self.min_adx) &
                (df['volume_ratio'] > self.volume_threshold)
        ).astype(int)

        # Exit signals
        df['short_exit'] = (
                (df['ma_fast'] > df['ma_slow']) |
                (df['rsi'] < self.rsi_oversold) |
                (df['macd_hist'] > 0)
        ).astype(int)

        df['long_exit'] = (
                (df['ma_fast'] < df['ma_slow']) |
                (df['rsi'] > self.rsi_overbought) |
                (df['macd_hist'] < 0)
        ).astype(int)

        # Track trailing values
        trailing_high = None
        trailing_low = None

        for i in range(50, len(df)):
            # Short entry (priority in bearish market)
            if df['short_signal'].iloc[i] == 1 and self.current_position is None:
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price + self.stop_atr * df['atr'].iloc[i]
                take_profit = entry_price - self.take_profit_atr * df['atr'].iloc[i]

                position_size = int((self.capital * self.risk_per_trade) / (stop_loss - entry_price))
                position_size = min(position_size, int(self.capital / entry_price))

                if position_size > 0:
                    self.current_position = {
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'direction': 'short',
                        'trailing_high': entry_price
                    }
                    trailing_high = entry_price

            # Long entry (only when market conditions are favorable)
            elif df['long_signal'].iloc[i] == 1 and self.current_position is None and self.use_long():
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - self.stop_atr * df['atr'].iloc[i]
                take_profit = entry_price + self.take_profit_atr * df['atr'].iloc[i]

                position_size = int((self.capital * self.risk_per_trade) / (entry_price - stop_loss))
                position_size = min(position_size, int(self.capital / entry_price))

                if position_size > 0:
                    self.current_position = {
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'direction': 'long',
                        'trailing_low': entry_price
                    }
                    trailing_low = entry_price

            # Exit logic
            elif self.current_position is not None:
                exit_signal = False
                exit_price = None
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]

                if self.current_position['direction'] == 'short':
                    # Update trailing high
                    trailing_high = max(trailing_high, current_high)
                    dynamic_stop = trailing_high - 0.7 * df['atr'].iloc[i]

                    if current_low <= self.current_position['take_profit']:
                        exit_price = self.current_position['take_profit']
                        exit_signal = True
                    elif current_high >= self.current_position['stop_loss']:
                        exit_price = self.current_position['stop_loss']
                        exit_signal = True
                    elif current_high >= dynamic_stop:
                        exit_price = dynamic_stop
                        exit_signal = True
                    elif df['short_exit'].iloc[i] == 1:
                        exit_price = df['close'].iloc[i]
                        exit_signal = True

                else:  # long
                    trailing_low = min(trailing_low, current_low)
                    dynamic_stop = trailing_low + 0.7 * df['atr'].iloc[i]

                    if current_high >= self.current_position['take_profit']:
                        exit_price = self.current_position['take_profit']
                        exit_signal = True
                    elif current_low <= self.current_position['stop_loss']:
                        exit_price = self.current_position['stop_loss']
                        exit_signal = True
                    elif current_low <= dynamic_stop:
                        exit_price = dynamic_stop
                        exit_signal = True
                    elif df['long_exit'].iloc[i] == 1:
                        exit_price = df['close'].iloc[i]
                        exit_signal = True

                if exit_signal:
                    if self.current_position['direction'] == 'short':
                        pnl = (self.current_position['entry_price'] - exit_price) * self.current_position['size']
                    else:
                        pnl = (exit_price - self.current_position['entry_price']) * self.current_position['size']

                    self.capital += pnl * (1 - self.commission)
                    self.positions.append({
                        **self.current_position,
                        'exit_time': df.index[i],
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    self.current_position = None
                    trailing_high = None
                    trailing_low = None

            self.equity_curve.append(self.capital)

        return self.calculate_metrics()

    def use_long(self):
        """Dynamic long bias based on market conditions"""
        if len(self.equity_curve) < 100:
            return False
        # Only go long if we have > 5% equity buffer
        return (self.capital / self.initial_capital) > 1.05


def run_systematic_optimization(df, param_ranges=None):
    """
    Run systematic grid search optimization to find the best parameters
    """
    if param_ranges is None:
        param_ranges = {
            'fast_ma': [5, 8, 10, 12, 15],
            'slow_ma': [15, 20, 21, 25, 30],
            'rsi_period': [10, 14, 20],
            'stop_atr': [1.0, 1.2, 1.5, 1.8, 2.0],
            'take_profit_atr': [1.8, 2.0, 2.2, 2.5, 3.0],
            'min_adx': [15, 20, 25],
            'risk_per_trade': [0.01, 0.015, 0.02, 0.025]
        }

    print("\n" + "=" * 80)
    print("SYSTEMATIC OPTIMIZATION STARTING")
    print("=" * 80)
    print(f"Parameter ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")
    print()

    # Calculate total combinations
    total_combinations = 1
    for values in param_ranges.values():
        total_combinations *= len(values)
    print(f"Total combinations to test: {total_combinations:,}")
    print()

    # Store all results
    all_results = []
    best_score = -np.inf
    best_params = None
    best_metrics = None

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    # Track progress
    tested = 0

    # Grid search through all combinations
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))

        # Ensure fast_ma < slow_ma
        if params['fast_ma'] >= params['slow_ma']:
            continue

        tested += 1

        # Test this parameter combination
        strategy = HybridOptimizedStrategy(
            name=f"Grid_{tested}",
            **params
        )

        # Progress indicator
        if tested % 50 == 0 or tested == 1:
            print(f"Testing combination {tested}/{total_combinations}...")

        metrics = strategy.backtest(df)

        # Calculate composite score (optimized for bearish markets)
        score = (
                metrics['total_return'] * 0.35 +  # Return is most important
                metrics['sharpe_ratio'] * 0.20 +  # Risk-adjusted returns
                (1 - abs(metrics['max_drawdown'])) * 0.20 +  # Lower drawdown is better
                metrics['profit_factor'] * 0.15 +  # Profitability
                metrics['win_rate'] * 0.10  # Win rate
        )

        # Store results
        all_results.append({
            'params': params,
            'metrics': metrics,
            'score': score
        })

        # Update best if found
        if score > best_score and metrics['num_trades'] >= 10:  # Minimum trades required
            best_score = score
            best_params = params
            best_metrics = metrics
            print(f"\n*** NEW BEST! Score: {score:.4f} ***")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Trades: {metrics['num_trades']}")

    # Sort results by score
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # Print top 10 results
    print("\n" + "=" * 80)
    print("TOP 10 BEST PARAMETER COMBINATIONS")
    print("=" * 80)

    for i, result in enumerate(all_results[:10], 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Parameters:")
        for param, value in result['params'].items():
            if param == 'risk_per_trade':
                print(f"     {param}: {value:.1%}")
            else:
                print(f"     {param}: {value}")
        print(f"   Performance:")
        print(f"     Total Return: {result['metrics']['total_return']:.2%}")
        print(f"     Sharpe Ratio: {result['metrics']['sharpe_ratio']:.3f}")
        print(f"     Max Drawdown: {result['metrics']['max_drawdown']:.2%}")
        print(f"     Win Rate: {result['metrics']['win_rate']:.2%}")
        print(f"     Profit Factor: {result['metrics']['profit_factor']:.2f}")
        print(f"     Num Trades: {result['metrics']['num_trades']}")

    return best_params, best_metrics, all_results


def save_best_params(best_params, filename="best_strategy_params.json"):
    """Save the best parameters to a file for later use"""
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params,
        'note': 'These parameters produced the best backtest results'
    }

    try:
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"\n✅ Best parameters saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving parameters: {e}")
        return False


def load_best_params(filename="best_strategy_params.json"):
    """Load previously saved best parameters"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('best_params')
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
    return None


def run_final_backtest(df, best_params):
    """Run final backtest with best parameters and detailed analysis"""
    print("\n" + "=" * 80)
    print("FINAL BACKTEST WITH BEST PARAMETERS")
    print("=" * 80)

    print(f"\nUsing parameters:")
    for param, value in best_params.items():
        if param == 'risk_per_trade':
            print(f"  {param}: {value:.1%}")
        else:
            print(f"  {param}: {value}")

    # Run final backtest
    strategy = HybridOptimizedStrategy(
        name="Best_Strategy",
        **best_params
    )

    metrics = strategy.backtest(df)

    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE RESULTS")
    print("=" * 80)

    print(f"\nRETURN METRICS:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {(1 + metrics['total_return']) ** (252 * 96 / len(df)) - 1:.2%}")
    print(f"  Avg Trade Return: {metrics['avg_trade_return']:.2%}")

    print(f"\nRISK METRICS:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  Recovery Factor: {metrics['recovery_factor']:.3f}")

    print(f"\nTRADE STATISTICS:")
    print(f"  Total Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Avg Duration: {metrics['avg_duration_hours']:.1f} hours")

    # Analyze trade distribution
    if len(strategy.positions) > 0:
        long_trades = [p for p in strategy.positions if p['direction'] == 'long']
        short_trades = [p for p in strategy.positions if p['direction'] == 'short']

        print(f"\nTRADE DISTRIBUTION:")
        print(f"  Long Trades: {len(long_trades)}")
        print(f"  Short Trades: {len(short_trades)}")

        if long_trades:
            long_returns = [(p['exit_price'] - p['entry_price']) / p['entry_price'] for p in long_trades]
            print(f"  Long Avg Return: {np.mean(long_returns):.2%}")
            print(f"  Long Win Rate: {sum(1 for r in long_returns if r > 0) / len(long_returns):.2%}")

        if short_trades:
            short_returns = [(p['entry_price'] - p['exit_price']) / p['entry_price'] for p in short_trades]
            print(f"  Short Avg Return: {np.mean(short_returns):.2%}")
            print(f"  Short Win Rate: {sum(1 for r in short_returns if r > 0) / len(short_returns):.2%}")

    # Show top 10 best trades
    print(f"\nTOP 10 BEST TRADES:")
    sorted_trades = sorted(strategy.positions, key=lambda x: x['pnl'], reverse=True)[:10]
    for i, trade in enumerate(sorted_trades, 1):
        ret = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        if trade['direction'] == 'short':
            ret = -ret
        print(
            f"  {i}. {trade['direction'].upper()} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | Return: {ret:.2%} | PnL: ${trade['pnl']:,.2f}")

    # Show worst 5 trades
    print(f"\nWORST 5 TRADES:")
    sorted_trades = sorted(strategy.positions, key=lambda x: x['pnl'])[:5]
    for i, trade in enumerate(sorted_trades, 1):
        ret = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        if trade['direction'] == 'short':
            ret = -ret
        print(
            f"  {i}. {trade['direction'].upper()} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | Return: {ret:.2%} | PnL: ${trade['pnl']:,.2f}")

    return strategy, metrics


# Main execution
print("\n" + "=" * 80)
print("COMPREHENSIVE OPTIMIZATION SYSTEM")
print("=" * 80)

# Check for existing best parameters
best_params = load_best_params()

if best_params:
    print("\n📁 Found saved best parameters from previous run!")
    print("Do you want to:")
    print("  1. Use saved parameters for final backtest")
    print("  2. Run new optimization (this will overwrite saved parameters)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        print("\n✅ Using saved best parameters for final backtest")
        run_final_backtest(df, best_params)
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)
        exit(0)

# Run systematic optimization
print("\n🔍 Running systematic grid search optimization...")
best_params, best_metrics, all_results = run_systematic_optimization(df)

# Save best parameters for future use
save_best_params(best_params)

# Run final backtest with best parameters
run_final_backtest(df, best_params)

# Show parameter convergence analysis
print("\n" + "=" * 80)
print("PARAMETER CONVERGENCE ANALYSIS")
print("=" * 80)

print("\nBest parameters found:")
for param, value in best_params.items():
    if param == 'risk_per_trade':
        print(f"  {param}: {value:.1%}")
    else:
        print(f"  {param}: {value}")

# Show how these compare to default values
default_params = {
    'fast_ma': 5,
    'slow_ma': 15,
    'rsi_period': 14,
    'stop_atr': 1.5,
    'take_profit_atr': 2.2,
    'min_adx': 20,
    'risk_per_trade': 0.02
}

print("\nComparison with default parameters:")
for param, best_val in best_params.items():
    default_val = default_params.get(param)
    if default_val:
        if param == 'risk_per_trade':
            change = (best_val - default_val) / default_val * 100
            print(f"  {param}: Best={best_val:.1%} | Default={default_val:.1%} | Change={change:+.1f}%")
        else:
            change = (best_val - default_val) / default_val * 100
            print(f"  {param}: Best={best_val} | Default={default_val} | Change={change:+.1f}%")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE - RESULTS SAVED")
print("=" * 80)