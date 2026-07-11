"""
Trading Signal Checker Program with Missed Opportunity Detection
Identifies when buy signals were missed and exits were premature
Includes comprehensive tabulation for decision making
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
from tabulate import tabulate
from colorama import Fore, Back, Style, init

# Initialize colorama for colored output
init(autoreset=True)


class TradingSignalChecker:
    def __init__(self, excel_file_path):
        """
        Initialize the trading signal checker with an Excel file

        Args:
            excel_file_path (str): Path to the Excel file
        """
        self.file_path = excel_file_path
        self.data = None
        self.current_row = None
        self.missed_opportunities_summary = None
        self.premature_exits_summary = None
        self.load_data()

    def load_data(self):
        """Load and validate the Excel file"""
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            # Load Excel file
            self.data = pd.read_excel(self.file_path)

            # Convert timestamp column to datetime if it's not already
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)

            # Check required columns
            required_columns = [
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'EMA_Fast', 'EMA_Mid', 'EMA_Slow', 'RSI', 'CCI', 'ADX',
                'ATR', 'Kalman_Strength', 'Volume_Ratio'
            ]

            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                print(f"⚠️ Warning: Missing columns: {missing_columns}")
                print("Program will continue but some checks may not work properly.")

            print(f"✅ Successfully loaded {len(self.data)} rows from {self.file_path}")
            print(f"📊 Columns found: {list(self.data.columns)}")

            # Display date range
            if 'timestamp' in self.data.columns:
                print(f"📅 Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise

    def display_latest_data(self, n=5):
        """Display the most recent n rows of data"""
        if self.data is not None:
            print(f"\n📈 Latest {n} rows of data:")
            print(self.data.tail(n).to_string())

    def find_row_by_datetime(self, target_datetime, tolerance_minutes=5):
        """
        Find the row closest to a specific datetime

        Args:
            target_datetime (datetime): The datetime to search for
            tolerance_minutes (int): Tolerance in minutes for finding close matches

        Returns:
            tuple: (index, row, distance_minutes) or (None, None, None) if not found
        """
        if 'timestamp' not in self.data.columns:
            print("❌ No timestamp column in data")
            return None, None, None

        # Calculate time difference for each row
        time_diffs = abs((self.data['timestamp'] - target_datetime).dt.total_seconds() / 60)

        # Find the closest row
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs[min_diff_idx]

        if min_diff <= tolerance_minutes:
            return min_diff_idx, self.data.iloc[min_diff_idx], min_diff
        else:
            return None, None, None

    def find_rows_in_date_range(self, start_datetime, end_datetime):
        """
        Find all rows within a specific date/time range

        Args:
            start_datetime (datetime): Start of range
            end_datetime (datetime): End of range

        Returns:
            DataFrame: Filtered data within the range
        """
        if 'timestamp' not in self.data.columns:
            print("❌ No timestamp column in data")
            return pd.DataFrame()

        mask = (self.data['timestamp'] >= start_datetime) & (self.data['timestamp'] <= end_datetime)
        filtered_data = self.data.loc[mask].copy()

        return filtered_data

    def check_entry_condition(self, row):
        """
        Check if entry conditions are met for a given row

        Args:
            row: pandas Series containing the data row

        Returns:
            tuple: (signal_type, reasons, confluence_score, bullish_count, bearish_count)
        """
        reasons = []
        confluence_score = 0

        try:
            # Check EMA alignment
            if row['EMA_Fast'] > row['EMA_Mid'] > row['EMA_Slow']:
                confluence_score += 20
                reasons.append("✅ EMA aligned bullishly (Fast > Mid > Slow)")
            elif row['EMA_Fast'] < row['EMA_Mid'] < row['EMA_Slow']:
                confluence_score += 20
                reasons.append("✅ EMA aligned bearishly (Fast < Mid < Slow)")
            else:
                reasons.append("❌ EMA not properly aligned")

            # Check RSI conditions
            if 30 <= row['RSI'] <= 70:
                confluence_score += 15
                reasons.append(f"✅ RSI in neutral zone ({row['RSI']:.1f})")
            elif row['RSI'] < 30:
                reasons.append(f"⚠️ RSI oversold ({row['RSI']:.1f}) - Potential buy setup")
            elif row['RSI'] > 70:
                reasons.append(f"⚠️ RSI overbought ({row['RSI']:.1f}) - Potential sell setup")

            # Check CCI
            if -100 <= row['CCI'] <= 100:
                confluence_score += 10
                reasons.append(f"✅ CCI neutral ({row['CCI']:.1f})")
            elif row['CCI'] < -100:
                reasons.append(f"⚠️ CCI oversold ({row['CCI']:.1f}) - Potential buy setup")
            elif row['CCI'] > 100:
                reasons.append(f"⚠️ CCI overbought ({row['CCI']:.1f}) - Potential sell setup")

            # Check ADX (trend strength)
            if row['ADX'] > 25:
                confluence_score += 15
                reasons.append(f"✅ Strong trend (ADX: {row['ADX']:.1f})")
            elif row['ADX'] > 20:
                reasons.append(f"📊 Moderate trend (ADX: {row['ADX']:.1f})")
            else:
                reasons.append(f"⚠️ Weak trend (ADX: {row['ADX']:.1f})")

            # Check Kalman Strength
            if abs(row['Kalman_Strength']) > 0.5:
                confluence_score += 15
                if row['Kalman_Strength'] > 0:
                    reasons.append(f"✅ Strong bullish Kalman signal ({row['Kalman_Strength']:.3f})")
                else:
                    reasons.append(f"✅ Strong bearish Kalman signal ({row['Kalman_Strength']:.3f})")
            else:
                reasons.append(f"⚠️ Weak Kalman signal ({row['Kalman_Strength']:.3f})")

            # Check Volume Ratio
            if row['Volume_Ratio'] > 1.2:
                confluence_score += 15
                reasons.append(f"✅ High volume confirmation ({row['Volume_Ratio']:.2f}x)")
            elif row['Volume_Ratio'] < 0.8:
                reasons.append(f"⚠️ Low volume ({row['Volume_Ratio']:.2f}x) - Weak confirmation")
            else:
                reasons.append(f"📊 Normal volume ({row['Volume_Ratio']:.2f}x)")

            # Price position relative to EMAs
            if row['Close'] > row['EMA_Fast']:
                confluence_score += 10
                reasons.append("✅ Price above fast EMA (bullish)")
            else:
                reasons.append("❌ Price below fast EMA (bearish)")

            # Determine signal type based on confluence and conditions
            signal_type = None

            # Bullish conditions
            bullish_count = 0
            if row['EMA_Fast'] > row['EMA_Mid']:
                bullish_count += 1
            if row['RSI'] > 50 and row['RSI'] < 70:
                bullish_count += 1
            if row['CCI'] > 0:
                bullish_count += 1
            if row['Kalman_Strength'] > 0.3:
                bullish_count += 1
            if row['Close'] > row['EMA_Fast']:
                bullish_count += 1

            # Bearish conditions
            bearish_count = 0
            if row['EMA_Fast'] < row['EMA_Mid']:
                bearish_count += 1
            if row['RSI'] < 50 and row['RSI'] > 30:
                bearish_count += 1
            if row['CCI'] < 0:
                bearish_count += 1
            if row['Kalman_Strength'] < -0.3:
                bearish_count += 1
            if row['Close'] < row['EMA_Fast']:
                bearish_count += 1

            # Determine signal based on counts and confluence
            if confluence_score >= 70:
                if bullish_count >= 4:
                    signal_type = "BUY"
                    reasons.append("📈 STRONG BUY signal - Multiple bullish conditions met")
                elif bearish_count >= 4:
                    signal_type = "SELL"
                    reasons.append("📉 STRONG SELL signal - Multiple bearish conditions met")
                elif bullish_count > bearish_count:
                    signal_type = "WEAK BUY"
                    reasons.append("📈 WEAK BUY signal - Some bullish conditions present")
                elif bearish_count > bullish_count:
                    signal_type = "WEAK SELL"
                    reasons.append("📉 WEAK SELL signal - Some bearish conditions present")

            return signal_type, reasons, confluence_score, bullish_count, bearish_count

        except Exception as e:
            return None, [f"Error checking conditions: {e}"], 0, 0, 0

    def check_exit_condition(self, row, position_type):
        """
        Check if exit conditions are met for an open position

        Args:
            row: pandas Series containing the data row
            position_type: 'BUY' or 'SELL'

        Returns:
            tuple: (should_exit, reasons, exit_type)
        """
        reasons = []
        exit_type = None
        should_exit = False

        try:
            # Check for exit signals from data
            if 'Exit_Signal' in self.data.columns and pd.notna(row.get('Exit_Signal')):
                if row['Exit_Signal'] == 1:
                    should_exit = True
                    exit_type = "SIGNAL"
                    reasons.append(f"🚩 Exit signal triggered: {row.get('Exit_Reason', 'No reason provided')}")

            # Technical exit conditions based on position type
            if position_type == "BUY":
                # Exit long position conditions
                if row['RSI'] > 80:
                    should_exit = True
                    exit_type = "RSI_OVERBOUGHT"
                    reasons.append(f"⚠️ RSI extremely overbought ({row['RSI']:.1f}) - Exit long")
                elif row['RSI'] > 75:
                    reasons.append(f"⚠️ RSI overbought ({row['RSI']:.1f}) - Consider taking profits")

                if row['Close'] < row['EMA_Fast']:
                    should_exit = True
                    exit_type = "EMA_BREAK"
                    reasons.append(f"⚠️ Price broke below fast EMA - Exit long")

                if row['ADX'] < 20:
                    reasons.append(f"📊 Trend weakening (ADX: {row['ADX']:.1f}) - Consider exit")

                if row['Kalman_Strength'] < -0.3:
                    should_exit = True
                    exit_type = "KALMAN_REVERSAL"
                    reasons.append(f"⚠️ Kalman turned bearish ({row['Kalman_Strength']:.3f}) - Exit long")

            elif position_type == "SELL":
                # Exit short position conditions
                if row['RSI'] < 20:
                    should_exit = True
                    exit_type = "RSI_OVERSOLD"
                    reasons.append(f"⚠️ RSI extremely oversold ({row['RSI']:.1f}) - Exit short")
                elif row['RSI'] < 25:
                    reasons.append(f"⚠️ RSI oversold ({row['RSI']:.1f}) - Consider covering shorts")

                if row['Close'] > row['EMA_Fast']:
                    should_exit = True
                    exit_type = "EMA_BREAK"
                    reasons.append(f"⚠️ Price broke above fast EMA - Exit short")

                if row['ADX'] < 20:
                    reasons.append(f"📊 Trend weakening (ADX: {row['ADX']:.1f}) - Consider exit")

                if row['Kalman_Strength'] > 0.3:
                    should_exit = True
                    exit_type = "KALMAN_REVERSAL"
                    reasons.append(f"⚠️ Kalman turned bullish ({row['Kalman_Strength']:.3f}) - Exit short")

            return should_exit, reasons, exit_type

        except Exception as e:
            return False, [f"Error checking exit conditions: {e}"], None

    def detect_missed_buy_opportunities(self, lookahead_bars=10, min_price_increase_pct=2.0):
        """
        Detect when buy signals were present but not taken, and price continued bullish

        Args:
            lookahead_bars (int): Number of bars to look ahead to check price movement
            min_price_increase_pct (float): Minimum price increase to consider as bullish continuation

        Returns:
            DataFrame: Missed opportunities with details
        """
        print("\n" + "=" * 70)
        print(f"{Fore.YELLOW}🔍 DETECTING MISSED BUY OPPORTUNITIES{Style.RESET_ALL}")
        print("=" * 70)
        print(f"Looking for buy signals that were missed while price continued bullish")
        print(f"Parameters: Look ahead {lookahead_bars} bars, Min increase {min_price_increase_pct}%")

        missed_opportunities = []

        for idx in range(len(self.data) - lookahead_bars):
            current_row = self.data.iloc[idx]

            # Check if this was a buy signal according to our conditions
            signal_type, reasons, confluence, bullish_count, bearish_count = self.check_entry_condition(current_row)

            # Check if it was a valid buy signal
            is_buy_signal = signal_type in ["BUY", "WEAK BUY"] and confluence >= 70

            if is_buy_signal:
                # Check if there was actually a buy execution in the data
                actual_buy_executed = False
                prevention_reason = "Unknown"

                # Check if there's an entry signal column
                if 'Entry_Signal' in self.data.columns:
                    if pd.notna(current_row.get('Entry_Signal')) and current_row['Entry_Signal'] == 1:
                        actual_buy_executed = True

                # If no entry signal column or no buy executed, check if it was prevented
                if not actual_buy_executed:
                    # Check for reasons why buy might have been prevented
                    prevention_reasons = []

                    # Check if RSI was overbought (could prevent buy)
                    if current_row['RSI'] > 70:
                        prevention_reasons.append(f"RSI overbought ({current_row['RSI']:.1f})")

                    # Check if price was too far from EMA
                    if abs(current_row['Close'] - current_row['EMA_Fast']) / current_row['EMA_Fast'] > 0.03:
                        prevention_reasons.append(
                            f"Price too extended from EMA ({((current_row['Close'] - current_row['EMA_Fast']) / current_row['EMA_Fast'] * 100):.2f}%)")

                    # Check if volume was low
                    if current_row['Volume_Ratio'] < 0.8:
                        prevention_reasons.append(f"Low volume ({current_row['Volume_Ratio']:.2f}x)")

                    # Check if ADX was weak
                    if current_row['ADX'] < 20:
                        prevention_reasons.append(f"Weak trend (ADX: {current_row['ADX']:.1f})")

                    # Check if Kalman strength was weak
                    if abs(current_row['Kalman_Strength']) < 0.3:
                        prevention_reasons.append(f"Weak Kalman signal ({current_row['Kalman_Strength']:.3f})")

                    prevention_reason = " | ".join(
                        prevention_reasons) if prevention_reasons else "No clear prevention reason"

                    # Now check if price continued bullish after this point
                    future_prices = self.data.iloc[idx + 1:idx + lookahead_bars + 1]['Close'].values
                    current_price = current_row['Close']

                    # Calculate maximum price increase
                    if len(future_prices) > 0:
                        max_future_price = max(future_prices)
                        price_increase_pct = ((max_future_price - current_price) / current_price) * 100

                        # Check if price increased significantly
                        if price_increase_pct >= min_price_increase_pct:
                            # Find when the price peaked
                            peak_idx = idx + 1 + np.argmax(future_prices)
                            peak_row = self.data.iloc[peak_idx]
                            peak_time = peak_row['timestamp']

                            missed_opportunities.append({
                                'timestamp': current_row['timestamp'],
                                'row_index': idx,
                                'price_at_signal': current_price,
                                'peak_price': max_future_price,
                                'price_increase_pct': price_increase_pct,
                                'peak_timestamp': peak_time,
                                'peak_row_index': peak_idx,
                                'confluence_score': confluence,
                                'prevention_reason': prevention_reason,
                                'rsi': current_row['RSI'],
                                'adx': current_row['ADX'],
                                'kalman_strength': current_row['Kalman_Strength'],
                                'volume_ratio': current_row['Volume_Ratio'],
                                'bullish_count': bullish_count,
                                'signal_type': signal_type
                            })

        # Create DataFrame of missed opportunities
        missed_df = pd.DataFrame(missed_opportunities)
        self.missed_opportunities_summary = missed_df

        if len(missed_df) > 0:
            missed_df = missed_df.sort_values('price_increase_pct', ascending=False)

            print(f"\n✅ Found {len(missed_df)} missed buy opportunities where price continued bullish")

            # Display summary table
            self.display_missed_opportunities_table(missed_df)

        else:
            print("\n❌ No missed buy opportunities found with current parameters")

        return missed_df

    def display_missed_opportunities_table(self, missed_df):
        """Display missed opportunities in a formatted table"""
        if len(missed_df) == 0:
            return

        print(f"\n{Fore.YELLOW}{'=' * 100}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 MISSED BUY OPPORTUNITIES SUMMARY TABLE{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 100}{Style.RESET_ALL}")

        # Prepare table data
        table_data = []
        for i, (_, opp) in enumerate(missed_df.head(20).iterrows(), 1):
            # Color code based on missed gain percentage
            if opp['price_increase_pct'] >= 10:
                gain_color = Fore.RED
            elif opp['price_increase_pct'] >= 5:
                gain_color = Fore.YELLOW
            else:
                gain_color = Fore.GREEN

            table_data.append([
                i,
                opp['timestamp'].strftime('%Y-%m-%d %H:%M'),
                opp['row_index'],
                f"${opp['price_at_signal']:.4f}",
                f"${opp['peak_price']:.4f}",
                f"{gain_color}{opp['price_increase_pct']:.2f}%{Style.RESET_ALL}",
                opp['peak_timestamp'].strftime('%Y-%m-%d %H:%M'),
                opp['peak_row_index'],
                opp['confluence_score'],
                f"{opp['rsi']:.1f}",
                opp['signal_type']
            ])

        # Define headers
        headers = [
            '#', 'Timestamp', 'Row', 'Price', 'Peak Price',
            'Missed Gain %', 'Peak Time', 'Peak Row', 'Confluence', 'RSI', 'Signal'
        ]

        # Print table
        print(tabulate(table_data, headers=headers, tablefmt='grid', numalign='right'))

        # Print statistics
        print(f"\n{Fore.CYAN}📈 Missed Opportunity Statistics:{Style.RESET_ALL}")
        print(f"   Total missed opportunities: {len(missed_df)}")
        print(f"   Average missed gain: {missed_df['price_increase_pct'].mean():.2f}%")
        print(f"   Max missed gain: {missed_df['price_increase_pct'].max():.2f}%")
        print(f"   Min missed gain: {missed_df['price_increase_pct'].min():.2f}%")
        print(
            f"   Total potential profit missed: ${(missed_df['peak_price'] - missed_df['price_at_signal']).sum():.2f}")

        # Group by prevention reason
        print(f"\n{Fore.CYAN}🔍 Missed Opportunities by Prevention Reason:{Style.RESET_ALL}")
        reason_counts = missed_df['prevention_reason'].value_counts().head(5)
        for reason, count in reason_counts.items():
            # Truncate long reasons
            short_reason = reason[:50] + "..." if len(reason) > 50 else reason
            print(f"   • {short_reason}: {count} opportunities")

    def detect_premature_exits(self, lookahead_bars=10, min_continuation_pct=2.0):
        """
        Detect when exit signals were executed but price continued bullish

        Args:
            lookahead_bars (int): Number of bars to look ahead after exit
            min_continuation_pct (float): Minimum price increase to consider as premature exit

        Returns:
            DataFrame: Premature exits with details
        """
        print("\n" + "=" * 70)
        print(f"{Fore.YELLOW}🔍 DETECTING PREMATURE EXITS{Style.RESET_ALL}")
        print("=" * 70)
        print(f"Looking for exits that were executed while price continued bullish")
        print(f"Parameters: Look ahead {lookahead_bars} bars, Min continuation {min_continuation_pct}%")

        premature_exits = []

        for idx in range(len(self.data) - lookahead_bars):
            current_row = self.data.iloc[idx]

            # Check if there was an exit signal at this point
            exit_occurred = False
            exit_reason = None

            # Check Exit_Signal column
            if 'Exit_Signal' in self.data.columns and pd.notna(current_row.get('Exit_Signal')):
                if current_row['Exit_Signal'] == 1:
                    exit_occurred = True
                    exit_reason = current_row.get('Exit_Reason', 'Signal triggered')

            # Also check technical exit conditions
            exit_long, exit_long_reasons, exit_long_type = self.check_exit_condition(current_row, "BUY")

            if exit_occurred or exit_long:
                # Check if price continued bullish after exit
                future_prices = self.data.iloc[idx + 1:idx + lookahead_bars + 1]['Close'].values
                current_price = current_row['Close']

                if len(future_prices) > 0:
                    max_future_price = max(future_prices)
                    price_increase_pct = ((max_future_price - current_price) / current_price) * 100

                    # If price increased significantly after exit, it was premature
                    if price_increase_pct >= min_continuation_pct:
                        # Find when the price peaked
                        peak_idx = idx + 1 + np.argmax(future_prices)
                        peak_row = self.data.iloc[peak_idx]
                        peak_time = peak_row['timestamp']

                        # Determine exit reason
                        actual_exit_reason = exit_reason or (exit_long_type if exit_long else "Technical exit")

                        # Check indicators at exit
                        premature_exits.append({
                            'timestamp': current_row['timestamp'],
                            'row_index': idx,
                            'exit_price': current_price,
                            'peak_price': max_future_price,
                            'missed_gain_pct': price_increase_pct,
                            'peak_timestamp': peak_time,
                            'peak_row_index': peak_idx,
                            'exit_reason': actual_exit_reason,
                            'rsi_at_exit': current_row['RSI'],
                            'adx_at_exit': current_row['ADX'],
                            'kalman_at_exit': current_row['Kalman_Strength'],
                            'price_at_exit': current_price,
                            'exit_type': 'EXIT_SIGNAL' if exit_occurred else 'TECHNICAL'
                        })

        # Create DataFrame of premature exits
        premature_df = pd.DataFrame(premature_exits)
        self.premature_exits_summary = premature_df

        if len(premature_df) > 0:
            premature_df = premature_df.sort_values('missed_gain_pct', ascending=False)

            print(f"\n✅ Found {len(premature_df)} premature exits where price continued bullish")

            # Display summary table
            self.display_premature_exits_table(premature_df)

        else:
            print("\n❌ No premature exits found with current parameters")

        return premature_df

    def display_premature_exits_table(self, premature_df):
        """Display premature exits in a formatted table"""
        if len(premature_df) == 0:
            return

        print(f"\n{Fore.YELLOW}{'=' * 110}{Style.RESET_ALL}")
        print(f"{Fore.RED}📊 PREMATURE EXITS SUMMARY TABLE{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 110}{Style.RESET_ALL}")

        # Prepare table data
        table_data = []
        for i, (_, exit_) in enumerate(premature_df.head(20).iterrows(), 1):
            # Color code based on missed gain percentage
            if exit_['missed_gain_pct'] >= 10:
                gain_color = Fore.RED
            elif exit_['missed_gain_pct'] >= 5:
                gain_color = Fore.YELLOW
            else:
                gain_color = Fore.GREEN

            table_data.append([
                i,
                exit_['timestamp'].strftime('%Y-%m-%d %H:%M'),
                exit_['row_index'],
                f"${exit_['exit_price']:.4f}",
                f"${exit_['peak_price']:.4f}",
                f"{gain_color}{exit_['missed_gain_pct']:.2f}%{Style.RESET_ALL}",
                exit_['peak_timestamp'].strftime('%Y-%m-%d %H:%M'),
                exit_['peak_row_index'],
                exit_['exit_reason'][:30] + "..." if len(exit_['exit_reason']) > 30 else exit_['exit_reason'],
                f"{exit_['rsi_at_exit']:.1f}"
            ])

        # Define headers
        headers = [
            '#', 'Exit Time', 'Row', 'Exit Price', 'Peak Price',
            'Missed Gain %', 'Peak Time', 'Peak Row', 'Exit Reason', 'RSI'
        ]

        # Print table
        print(tabulate(table_data, headers=headers, tablefmt='grid', numalign='right'))

        # Print statistics
        print(f"\n{Fore.CYAN}📈 Premature Exit Statistics:{Style.RESET_ALL}")
        print(f"   Total premature exits: {len(premature_df)}")
        print(f"   Average missed gain: {premature_df['missed_gain_pct'].mean():.2f}%")
        print(f"   Max missed gain: {premature_df['missed_gain_pct'].max():.2f}%")
        print(f"   Min missed gain: {premature_df['missed_gain_pct'].min():.2f}%")
        print(f"   Total profit left on table: ${(premature_df['peak_price'] - premature_df['exit_price']).sum():.2f}")

        # Group by exit reason
        print(f"\n{Fore.CYAN}🔍 Premature Exits by Reason:{Style.RESET_ALL}")
        reason_counts = premature_df['exit_reason'].value_counts().head(5)
        for reason, count in reason_counts.items():
            short_reason = reason[:40] + "..." if len(reason) > 40 else reason
            print(f"   • {short_reason}: {count} exits")

    def generate_comprehensive_report(self):
        """Generate a comprehensive decision-making report with all findings"""
        print("\n" + "=" * 100)
        print(f"{Fore.MAGENTA}{Style.BRIGHT}📋 COMPREHENSIVE TRADING DECISION REPORT{Style.RESET_ALL}")
        print("=" * 100)

        # Section 1: Overall Statistics
        print(f"\n{Fore.CYAN}{Style.BRIGHT}1. OVERALL STATISTICS{Style.RESET_ALL}")
        print("-" * 50)
        print(f"   Total data rows: {len(self.data)}")
        print(f"   Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"   Price range: ${self.data['Close'].min():.4f} - ${self.data['Close'].max():.4f}")
        print(f"   Average price: ${self.data['Close'].mean():.4f}")

        # Section 2: Missed Opportunities Summary
        if self.missed_opportunities_summary is not None and len(self.missed_opportunities_summary) > 0:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}2. MISSED BUY OPPORTUNITIES SUMMARY{Style.RESET_ALL}")
            print("-" * 50)

            missed_df = self.missed_opportunities_summary

            # Top 5 most costly missed opportunities
            print(f"\n   {Fore.YELLOW}Top 5 Most Costly Missed Opportunities:{Style.RESET_ALL}")
            top_missed = missed_df.nlargest(5, 'price_increase_pct')[
                ['timestamp', 'price_at_signal', 'peak_price', 'price_increase_pct', 'prevention_reason']]
            for i, (_, opp) in enumerate(top_missed.iterrows(), 1):
                print(f"   {i}. {opp['timestamp']} - Missed {opp['price_increase_pct']:.2f}% gain")
                print(f"      Price: ${opp['price_at_signal']:.4f} → ${opp['peak_price']:.4f}")
                print(f"      Prevention: {opp['prevention_reason'][:60]}")

            # Prevention reason analysis
            print(f"\n   {Fore.YELLOW}Prevention Reason Analysis:{Style.RESET_ALL}")
            reason_stats = missed_df['prevention_reason'].value_counts()
            total_missed = len(missed_df)
            for reason, count in reason_stats.head(5).items():
                percentage = (count / total_missed) * 100
                short_reason = reason[:50] + "..." if len(reason) > 50 else reason
                print(f"   • {short_reason}: {count} opportunities ({percentage:.1f}%)")

            # Recommendation based on prevention reasons
            print(f"\n   {Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
            if 'RSI overbought' in str(reason_stats.index):
                print(
                    f"   • ⚠️ RSI overbought is preventing entries - Consider relaxing RSI threshold in strong trends")
            if 'Low volume' in str(reason_stats.index):
                print(f"   • 📊 Low volume preventing entries - Volume may be less critical in established trends")
            if 'Weak trend' in str(reason_stats.index):
                print(
                    f"   • 📈 ADX < 20 preventing entries - Consider allowing entries during trend establishment phase")

        # Section 3: Premature Exits Summary
        if self.premature_exits_summary is not None and len(self.premature_exits_summary) > 0:
            print(f"\n{Fore.RED}{Style.BRIGHT}3. PREMATURE EXITS SUMMARY{Style.RESET_ALL}")
            print("-" * 50)

            premature_df = self.premature_exits_summary

            # Top 5 most costly premature exits
            print(f"\n   {Fore.YELLOW}Top 5 Most Costly Premature Exits:{Style.RESET_ALL}")
            top_premature = premature_df.nlargest(5, 'missed_gain_pct')[
                ['timestamp', 'exit_price', 'peak_price', 'missed_gain_pct', 'exit_reason']]
            for i, (_, exit_) in enumerate(top_premature.iterrows(), 1):
                print(f"   {i}. {exit_['timestamp']} - Left {exit_['missed_gain_pct']:.2f}% on table")
                print(f"      Exit: ${exit_['exit_price']:.4f} → Peak: ${exit_['peak_price']:.4f}")
                print(f"      Exit reason: {exit_['exit_reason'][:60]}")

            # Exit reason analysis
            print(f"\n   {Fore.YELLOW}Exit Reason Analysis:{Style.RESET_ALL}")
            exit_stats = premature_df['exit_reason'].value_counts()
            total_exits = len(premature_df)
            for reason, count in exit_stats.head(5).items():
                percentage = (count / total_exits) * 100
                short_reason = reason[:50] + "..." if len(reason) > 50 else reason
                print(f"   • {short_reason}: {count} exits ({percentage:.1f}%)")

            # Recommendations based on exit reasons
            print(f"\n   {Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
            if 'RSI_OVERBOUGHT' in str(exit_stats.index):
                print(
                    f"   • ⚠️ RSI overbought causing premature exits - Consider trailing stops instead of fixed RSI exits")
            if 'EMA_BREAK' in str(exit_stats.index):
                print(f"   • 📉 EMA breaks causing premature exits - Use multiple timeframe confirmation")
            if 'KALMAN_REVERSAL' in str(exit_stats.index):
                print(f"   • 🔄 Kalman reversals premature - Add confirmation from other indicators")

        # Section 4: Combined Analysis & Action Items
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}4. ACTION ITEMS & STRATEGY ADJUSTMENTS{Style.RESET_ALL}")
        print("-" * 50)

        action_items = []

        # Generate action items based on findings
        if self.missed_opportunities_summary is not None and len(self.missed_opportunities_summary) > 0:
            avg_missed_gain = self.missed_opportunities_summary['price_increase_pct'].mean()
            if avg_missed_gain > 5:
                action_items.append(
                    f"🔴 HIGH PRIORITY: Average missed gain of {avg_missed_gain:.2f}% - Review entry criteria immediately")

        if self.premature_exits_summary is not None and len(self.premature_exits_summary) > 0:
            avg_premature_loss = self.premature_exits_summary['missed_gain_pct'].mean()
            total_left = (self.premature_exits_summary['peak_price'] - self.premature_exits_summary['exit_price']).sum()
            if avg_premature_loss > 3:
                action_items.append(
                    f"🟡 MEDIUM PRIORITY: Average {avg_premature_loss:.2f}% left on table from premature exits (Total: ${total_left:.2f})")

        # Specific strategy adjustments
        if self.missed_opportunities_summary is not None and len(self.missed_opportunities_summary) > 0:
            rsi_prevented = self.missed_opportunities_summary[
                self.missed_opportunities_summary['prevention_reason'].str.contains('RSI', na=False)]
            if len(rsi_prevented) > 0:
                action_items.append(
                    f"📊 Consider relaxing RSI entry threshold in strong trends ({len(rsi_prevented)} opportunities missed)")

            volume_prevented = self.missed_opportunities_summary[
                self.missed_opportunities_summary['prevention_reason'].str.contains('volume', na=False)]
            if len(volume_prevented) > 0:
                action_items.append(
                    f"📈 Volume requirement may be too strict - {len(volume_prevented)} opportunities missed due to low volume")

        if len(action_items) > 0:
            for item in action_items:
                print(f"   {item}")
        else:
            print(f"   ✅ No critical action items identified - Strategy performing well")

        # Section 5: Performance Metrics
        print(f"\n{Fore.BLUE}{Style.BRIGHT}5. STRATEGY PERFORMANCE METRICS{Style.RESET_ALL}")
        print("-" * 50)

        # Calculate metrics
        total_potential_missed = 0
        if self.missed_opportunities_summary is not None:
            total_potential_missed = (
                        self.missed_opportunities_summary['peak_price'] - self.missed_opportunities_summary[
                    'price_at_signal']).sum()

        total_premature_loss = 0
        if self.premature_exits_summary is not None:
            total_premature_loss = (
                        self.premature_exits_summary['peak_price'] - self.premature_exits_summary['exit_price']).sum()

        print(f"   Total potential profit from missed entries: ${total_potential_missed:.2f}")
        print(f"   Total profit left on table from premature exits: ${total_premature_loss:.2f}")
        print(f"   Combined opportunity cost: ${total_potential_missed + total_premature_loss:.2f}")

        if total_potential_missed + total_premature_loss > 0:
            print(f"\n   {Fore.YELLOW}Opportunity Cost Breakdown:{Style.RESET_ALL}")
            if total_potential_missed > 0:
                pct_missed = (total_potential_missed / (total_potential_missed + total_premature_loss)) * 100
                print(f"   • Missed entries: {pct_missed:.1f}% of total opportunity cost")
            if total_premature_loss > 0:
                pct_premature = (total_premature_loss / (total_potential_missed + total_premature_loss)) * 100
                print(f"   • Premature exits: {pct_premature:.1f}% of total opportunity cost")

        print("\n" + "=" * 100)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}📌 REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 100)

    def analyze_row(self, row_index, show_details=True):
        """
        Analyze a specific row by index

        Args:
            row_index (int): Index of the row to analyze
            show_details (bool): Whether to show detailed output
        """
        if row_index < 0 or row_index >= len(self.data):
            print(f"❌ Invalid row index. Please choose between 0 and {len(self.data) - 1}")
            return None

        row = self.data.iloc[row_index]

        if show_details:
            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS FOR ROW {row_index}")
            print(f"Timestamp: {row.get('timestamp', 'N/A')}")
            print("=" * 60)

            # Display key price data
            print(f"\n💰 Price Data:")
            print(f"  Open: {row.get('Open', 'N/A'):.4f}")
            print(f"  High: {row.get('High', 'N/A'):.4f}")
            print(f"  Low: {row.get('Low', 'N/A'):.4f}")
            print(f"  Close: {row.get('Close', 'N/A'):.4f}")
            print(f"  Volume: {row.get('Volume', 'N/A'):.0f}")

            # Display indicators
            print(f"\n📈 Technical Indicators:")
            print(f"  EMA Fast: {row.get('EMA_Fast', 'N/A'):.4f}")
            print(f"  EMA Mid: {row.get('EMA_Mid', 'N/A'):.4f}")
            print(f"  EMA Slow: {row.get('EMA_Slow', 'N/A'):.4f}")
            print(f"  RSI: {row.get('RSI', 'N/A'):.2f}")
            print(f"  CCI: {row.get('CCI', 'N/A'):.2f}")
            print(f"  ADX: {row.get('ADX', 'N/A'):.2f}")
            print(f"  ATR: {row.get('ATR', 'N/A'):.4f}")
            print(f"  Kalman Strength: {row.get('Kalman_Strength', 'N/A'):.4f}")
            print(f"  Volume Ratio: {row.get('Volume_Ratio', 'N/A'):.2f}")

        # Check entry conditions
        signal_type, reasons, confluence, bullish_count, bearish_count = self.check_entry_condition(row)

        if show_details:
            print(f"\n🎯 ENTRY ANALYSIS:")
            for reason in reasons:
                print(f"  {reason}")

            print(f"\n📊 Confluence Score: {confluence}/100")
            print(f"  Bullish signals: {bullish_count}/5")
            print(f"  Bearish signals: {bearish_count}/5")

            if signal_type:
                print(f"\n✅ TRADING SIGNAL: {signal_type}")

                # Show risk allocation if available
                if 'Risk_Allocation_%' in self.data.columns and pd.notna(row.get('Risk_Allocation_%')):
                    print(f"📊 Recommended Risk Allocation: {row['Risk_Allocation_%']}%")

                # Check if this matches existing signal
                if 'Entry_Signal' in self.data.columns:
                    existing_signal = row.get('Entry_Signal')
                    if pd.notna(existing_signal):
                        print(f"📌 Existing Signal in data: {existing_signal}")
                        if (existing_signal == 1 and signal_type in ["BUY", "WEAK BUY"]) or \
                                (existing_signal == -1 and signal_type in ["SELL", "WEAK SELL"]):
                            print("✅ Consistent with existing signal")
                        else:
                            print("⚠️ Different from existing signal - MISSED OPPORTUNITY!")
            else:
                print(f"\n❌ NO CLEAR TRADING SIGNAL")
                if confluence >= 50:
                    print("   Some conditions met but not enough for clear signal")
                elif confluence >= 30:
                    print("   Few conditions met - not enough for trading")
                else:
                    print("   Insufficient conditions for trading")

            # Check exit conditions
            print(f"\n🚪 EXIT ANALYSIS:")

            # Check long exit
            exit_long, long_reasons, exit_type = self.check_exit_condition(row, "BUY")
            print(f"\n  For LONG positions:")
            if long_reasons:
                for reason in long_reasons:
                    print(f"    {reason}")
            else:
                print("    No exit signals for long positions")
            if exit_long:
                print(f"    ✅ EXIT LONG signal: {exit_type}")

            # Check short exit
            exit_short, short_reasons, exit_type = self.check_exit_condition(row, "SELL")
            print(f"\n  For SHORT positions:")
            if short_reasons:
                for reason in short_reasons:
                    print(f"    {reason}")
            else:
                print("    No exit signals for short positions")
            if exit_short:
                print(f"    ✅ EXIT SHORT signal: {exit_type}")

            # Display existing exit info if available
            if 'Exit_Signal' in self.data.columns and pd.notna(row.get('Exit_Signal')):
                print(f"\n📌 Existing Exit Signal in data: {row['Exit_Signal']}")
                if pd.notna(row.get('Exit_Reason')):
                    print(f"   Reason: {row['Exit_Reason']}")

            print("\n" + "=" * 60)

        return {
            'row_index': row_index,
            'timestamp': row.get('timestamp'),
            'signal': signal_type,
            'confluence_score': confluence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'price': row.get('Close'),
            'exit_long': exit_long if show_details else None,
            'exit_short': exit_short if show_details else None
        }

    def analyze_date_range(self, start_datetime, end_datetime):
        """
        Analyze all rows within a date range

        Args:
            start_datetime (datetime): Start of range
            end_datetime (datetime): End of range
        """
        filtered_data = self.find_rows_in_date_range(start_datetime, end_datetime)

        if len(filtered_data) == 0:
            print(f"\n❌ No data found between {start_datetime} and {end_datetime}")
            return

        print("\n" + "=" * 70)
        print(f"📊 DATE RANGE ANALYSIS: {start_datetime} to {end_datetime}")
        print("=" * 70)
        print(f"Found {len(filtered_data)} rows in this range")

        # Analyze each row in the range
        signals_found = []

        for idx, row in filtered_data.iterrows():
            signal_type, reasons, confluence, bullish, bearish = self.check_entry_condition(row)

            if signal_type:
                signals_found.append({
                    'timestamp': row['timestamp'],
                    'signal': signal_type,
                    'confluence': confluence,
                    'price': row['Close'],
                    'row_index': idx
                })

        # Display summary
        if signals_found:
            print(f"\n✅ SIGNALS FOUND: {len(signals_found)}")
            print("\nSignal Details:")
            print("-" * 50)
            for signal in signals_found:
                print(f"📅 {signal['timestamp']}")
                print(f"   Signal: {signal['signal']}")
                print(f"   Price: {signal['price']:.4f}")
                print(f"   Confluence: {signal['confluence']}/100")
                print(f"   Row: {signal['row_index']}")
                print()
        else:
            print("\n❌ No clear trading signals found in this date range")

        # Calculate statistics
        avg_confluence = filtered_data.apply(lambda row: self.check_entry_condition(row)[2], axis=1).mean()
        print(f"\n📊 Range Statistics:")
        print(f"   Average Confluence Score: {avg_confluence:.1f}/100")
        print(f"   Highest Price: {filtered_data['Close'].max():.4f}")
        print(f"   Lowest Price: {filtered_data['Close'].min():.4f}")
        print(
            f"   Price Change: {((filtered_data['Close'].iloc[-1] - filtered_data['Close'].iloc[0]) / filtered_data['Close'].iloc[0] * 100):.2f}%")

    def search_by_datetime(self):
        """Search for a specific datetime"""
        print("\n🔍 SEARCH BY DATE AND TIME")
        print("-" * 40)

        # Get date input
        while True:
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                break
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")

        # Get time input (optional)
        time_str = input("Enter time (HH:MM, optional - press Enter to skip): ").strip()
        if time_str:
            try:
                time_obj = datetime.strptime(time_str, "%H:%M").time()
                target_datetime = datetime.combine(date_obj.date(), time_obj)
            except ValueError:
                print("❌ Invalid time format. Using 00:00")
                target_datetime = datetime.combine(date_obj.date(), datetime.min.time())
        else:
            target_datetime = datetime.combine(date_obj.date(), datetime.min.time())

        # Get tolerance
        tolerance_input = input("Enter tolerance in minutes (default 5): ").strip()
        tolerance = int(tolerance_input) if tolerance_input else 5

        # Search
        idx, row, distance = self.find_row_by_datetime(target_datetime, tolerance)

        if idx is not None:
            print(f"\n✅ Found closest match at {row['timestamp']} (distance: {distance:.1f} minutes)")
            self.analyze_row(idx)
        else:
            print(f"\n❌ No data found within {tolerance} minutes of {target_datetime}")

            # Show nearest available datetimes
            if 'timestamp' in self.data.columns:
                print("\nNearest available datetimes:")
                time_diffs = abs((self.data['timestamp'] - target_datetime).dt.total_seconds() / 3600)
                nearest_indices = time_diffs.nsmallest(3).index
                for i, idx in enumerate(nearest_indices, 1):
                    print(f"  {i}. {self.data.loc[idx, 'timestamp']} (diff: {time_diffs[idx]:.1f} hours)")

    def search_date_range(self):
        """Search within a date range"""
        print("\n🔍 SEARCH BY DATE RANGE")
        print("-" * 40)

        # Get start date
        while True:
            start_str = input("Enter start date (YYYY-MM-DD): ").strip()
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                break
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")

        # Get start time (optional)
        start_time_str = input("Enter start time (HH:MM, optional): ").strip()
        if start_time_str:
            try:
                start_time = datetime.strptime(start_time_str, "%H:%M").time()
                start_datetime = datetime.combine(start_date.date(), start_time)
            except ValueError:
                print("❌ Invalid time format. Using 00:00")
                start_datetime = datetime.combine(start_date.date(), datetime.min.time())
        else:
            start_datetime = datetime.combine(start_date.date(), datetime.min.time())

        # Get end date
        while True:
            end_str = input("Enter end date (YYYY-MM-DD): ").strip()
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                break
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")

        # Get end time (optional)
        end_time_str = input("Enter end time (HH:MM, optional): ").strip()
        if end_time_str:
            try:
                end_time = datetime.strptime(end_time_str, "%H:%M").time()
                end_datetime = datetime.combine(end_date.date(), end_time)
            except ValueError:
                print("❌ Invalid time format. Using 23:59")
                end_datetime = datetime.combine(end_date.date(), datetime.max.time())
        else:
            end_datetime = datetime.combine(end_date.date(), datetime.max.time())

        # Validate range
        if start_datetime > end_datetime:
            print("❌ Start date must be before end date")
            return

        # Analyze the range
        self.analyze_date_range(start_datetime, end_datetime)

    def find_best_signals(self, n=5):
        """Find the best buy and sell signals in the entire dataset"""
        print(f"\n🏆 TOP {n} SIGNALS IN DATASET")
        print("=" * 60)

        signals = []
        for idx, row in self.data.iterrows():
            signal_type, reasons, confluence, bullish, bearish = self.check_entry_condition(row)
            if signal_type:
                signals.append({
                    'timestamp': row['timestamp'],
                    'signal': signal_type,
                    'confluence': confluence,
                    'price': row['Close'],
                    'row_index': idx,
                    'bullish': bullish,
                    'bearish': bearish
                })

        # Sort by confluence score
        signals.sort(key=lambda x: x['confluence'], reverse=True)

        if signals:
            print(f"\nTop {min(n, len(signals))} Signals:")
            print("-" * 60)
            for i, signal in enumerate(signals[:n], 1):
                print(f"{i}. {signal['timestamp']}")
                print(f"   Signal: {signal['signal']}")
                print(f"   Price: {signal['price']:.4f}")
                print(f"   Confluence: {signal['confluence']}/100")
                print(f"   Bullish/Bearish: {signal['bullish']}/{signal['bearish']}")
                print()
        else:
            print("No signals found in dataset")

    def export_missed_opportunities(self, missed_df, filename="missed_opportunities.csv"):
        """Export missed opportunities to CSV"""
        if len(missed_df) > 0:
            missed_df.to_csv(filename, index=False)
            print(f"✅ Exported {len(missed_df)} missed opportunities to {filename}")

    def export_premature_exits(self, premature_df, filename="premature_exits.csv"):
        """Export premature exits to CSV"""
        if len(premature_df) > 0:
            premature_df.to_csv(filename, index=False)
            print(f"✅ Exported {len(premature_df)} premature exits to {filename}")

    def export_comprehensive_report(self, filename="trading_decision_report.txt"):
        """Export the comprehensive report to a text file"""
        import sys
        from io import StringIO

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        self.generate_comprehensive_report()

        # Get the report content
        report_content = sys.stdout.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        # Write to file
        with open(filename, 'w') as f:
            f.write(report_content)

        print(f"✅ Comprehensive report exported to {filename}")

    def interactive_mode(self):
        """Run interactive mode where user can analyze different rows"""
        print("\n" + "=" * 60)
        print("🚀 TRADING SIGNAL CHECKER - INTERACTIVE MODE")
        print("=" * 60)
        print(f"Loaded {len(self.data)} rows of data")
        if 'timestamp' in self.data.columns:
            print(f"Data range: {self.data['timestamp'].iloc[0]} to {self.data['timestamp'].iloc[-1]}")

        while True:
            print("\n" + "-" * 40)
            print("Options:")
            print("  [row_number]     - Analyze specific row by index")
            print("  [M]              - Detect MISSED BUY opportunities")
            print("  [P]              - Detect PREMATURE EXITS")
            print("  [RPT]            - Generate COMPREHENSIVE REPORT")
            print("  [D]              - Search by specific date/time")
            print("  [R]              - Search within date range")
            print("  [T]              - Find top signals")
            print("  [L]              - Show latest row")
            print("  [E]              - Show earliest row")
            print("  [rand]           - Show random row")
            print("  [Q]              - Quit")

            choice = input("\nEnter your choice: ").strip().upper()

            if choice == 'Q':
                print("Goodbye! 👋")
                break
            elif choice == 'L':
                self.analyze_row(len(self.data) - 1)
            elif choice == 'E':
                self.analyze_row(0)
            elif choice == 'RAND':
                self.analyze_row(random.randint(0, len(self.data) - 1))
            elif choice == 'D':
                self.search_by_datetime()
            elif choice == 'R':
                self.search_date_range()
            elif choice == 'T':
                try:
                    n = int(input("How many top signals to show? (default 5): ").strip() or "5")
                    self.find_best_signals(n)
                except ValueError:
                    self.find_best_signals(5)
            elif choice == 'M':
                try:
                    lookahead = int(input("Lookahead bars (default 10): ").strip() or "10")
                    min_pct = float(input("Minimum price increase % (default 2.0): ").strip() or "2.0")
                    missed_df = self.detect_missed_buy_opportunities(lookahead_bars=lookahead,
                                                                     min_price_increase_pct=min_pct)
                    if len(missed_df) > 0:
                        export = input("\nExport to CSV? (y/n): ").strip().upper()
                        if export == 'Y':
                            self.export_missed_opportunities(missed_df)
                except ValueError:
                    print("❌ Invalid input. Using defaults.")
                    missed_df = self.detect_missed_buy_opportunities()
            elif choice == 'P':
                try:
                    lookahead = int(input("Lookahead bars (default 10): ").strip() or "10")
                    min_pct = float(input("Minimum missed gain % (default 2.0): ").strip() or "2.0")
                    premature_df = self.detect_premature_exits(lookahead_bars=lookahead,
                                                               min_continuation_pct=min_pct)
                    if len(premature_df) > 0:
                        export = input("\nExport to CSV? (y/n): ").strip().upper()
                        if export == 'Y':
                            self.export_premature_exits(premature_df)
                except ValueError:
                    print("❌ Invalid input. Using defaults.")
                    premature_df = self.detect_premature_exits()
            elif choice == 'RPT':
                self.generate_comprehensive_report()
                export = input("\nExport report to file? (y/n): ").strip().upper()
                if export == 'Y':
                    self.export_comprehensive_report()
            else:
                try:
                    row_num = int(choice)
                    self.analyze_row(row_num)
                except ValueError:
                    print("❌ Invalid input. Please try again.")


def main():
    """Main function to run the program"""
    print("\n" + "=" * 60)
    print("📊 TRADING SIGNAL CHECKER v4.0")
    print("   with Comprehensive Decision Support")
    print("=" * 60)

    # Install required packages if not present
    try:
        from tabulate import tabulate
        from colorama import Fore, Back, Style, init
    except ImportError:
        print("Installing required packages...")
        os.system("pip install tabulate colorama")
        from tabulate import tabulate
        from colorama import Fore, Back, Style, init
        init(autoreset=True)

    # Get Excel file path from user
    while True:
        file_path = input("\nEnter the path to your Excel file: ").strip()

        # Remove quotes if user pasted with quotes
        file_path = file_path.strip('"').strip("'")

        if os.path.exists(file_path):
            break
        else:
            print(f"❌ File not found: {file_path}")
            print("Please check the path and try again.")

    try:
        # Initialize checker
        checker = TradingSignalChecker(file_path)

        # Show preview
        checker.display_latest_data()

        # Run interactive mode
        checker.interactive_mode()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return


if __name__ == "__main__":
    main()