#!/usr/bin/env python3
"""
AUTOMATED PARAMETER UPDATER - Step 1+2 Combined
================================================

This script automatically updates your momentum strategy parameters
for Step 1+2 (Quality + ADX optimization).

WHAT IT DOES:
- Makes a backup of your original file
- Updates 14 parameters automatically
- Verifies all changes were successful
- Creates a summary report

USAGE:
    python update_step1+2.py your_strategy_file.py

SAFETY:
- Original file is backed up with timestamp
- No changes unless all verifications pass
- You can rollback using the backup file
"""

import sys
import os
import re
from datetime import datetime
import shutil

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS TO UPDATE (Step 1+2)
# ═══════════════════════════════════════════════════════════════════════════

UPDATES = [
    # (parameter_name, old_value, new_value, description)
    ("quality_tier1_min", "70", "60", "Lower quality threshold for Tier 1"),
    ("quality_tier2_min", "82", "75", "Lower quality threshold for Tier 2"),
    ("tier1_adx_hard_min", "20", "15", "Catch early trend formation"),
    ("tier1_adx_min", "18", "15", "Lower ADX minimum for Tier 1"),
    ("adx_min", "18", "15", "Lower global ADX minimum"),
    ("adx_min_trend", "18", "15", "Lower ADX trend minimum"),
    ("adx_score_trend_forming", "18", "15", "Lower ADX scoring - trend forming"),
    ("adx_score_good_trend", "22", "20", "Lower ADX scoring - good trend"),
    ("adx_score_strong_trend", "28", "26", "Lower ADX scoring - strong trend"),
    ("adx_score_very_strong", "33", "31", "Lower ADX scoring - very strong"),
    ("adx_score_extended", "38", "36", "Lower ADX scoring - extended"),
    ("tier2_adx_min", "15", "12", "Lower ADX minimum for Tier 2"),
    ("fuzzy_absolute_min", "55", "50", "Lower fuzzy learning floor"),
    ("ema_near_tolerance", "0.008", "0.010", "Widen EMA tolerance"),
]


def backup_file(filepath):
    """Create timestamped backup of original file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_before_step1+2_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Backup created: {backup_path}")
    return backup_path


def read_file(filepath):
    """Read file contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath, content):
    """Write content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def update_parameter(content, param_name, old_value, new_value):
    """
    Update a single parameter in the content.
    Returns (updated_content, success, count)
    """
    # Pattern to match parameter assignments like:
    # "param_name": old_value,
    pattern = rf'("{param_name}"\s*:\s*)({old_value})(\s*,)'
    
    # Count how many times we'll replace
    count = len(re.findall(pattern, content))
    
    if count == 0:
        return content, False, 0
    
    # Replace all occurrences
    updated_content = re.sub(pattern, rf'\g<1>{new_value}\g<3>', content)
    
    return updated_content, True, count


def verify_updates(content, updates_applied):
    """Verify all updates were successful"""
    all_good = True
    issues = []
    
    for param_name, old_value, new_value, _ in UPDATES:
        # Check new value exists
        new_pattern = rf'"{param_name}"\s*:\s*{new_value}'
        new_matches = len(re.findall(new_pattern, content))
        
        # Check old value is gone
        old_pattern = rf'"{param_name}"\s*:\s*{old_value}'
        old_matches = len(re.findall(old_pattern, content))
        
        if new_matches == 0:
            all_good = False
            issues.append(f"  ✗ {param_name}: New value {new_value} not found!")
        elif old_matches > 0:
            all_good = False
            issues.append(f"  ✗ {param_name}: Old value {old_value} still present!")
        else:
            # Success - old gone, new present
            pass
    
    return all_good, issues


def main():
    """Main execution"""
    
    print("=" * 70)
    print("MOMENTUM STRATEGY - STEP 1+2 PARAMETER UPDATER")
    print("=" * 70)
    print()
    
    # Check command line argument
    if len(sys.argv) != 2:
        print("USAGE: python update_step1+2.py <your_strategy_file.py>")
        print()
        print("EXAMPLE:")
        print("  python update_step1+2.py momentum_v77.py")
        print()
        return 1
    
    filepath = sys.argv[1]
    
    # Validate file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return 1
    
    print(f"Target file: {filepath}")
    print()
    
    # Create backup
    print("STEP 1: Creating backup...")
    backup_path = backup_file(filepath)
    print()
    
    # Read original content
    print("STEP 2: Reading file...")
    content = read_file(filepath)
    print(f"✓ File read: {len(content)} bytes")
    print()
    
    # Apply updates
    print("STEP 3: Applying updates...")
    print("-" * 70)
    
    updated_content = content
    updates_applied = []
    
    for param_name, old_value, new_value, description in UPDATES:
        updated_content, success, count = update_parameter(
            updated_content, param_name, old_value, new_value
        )
        
        status = "✓" if success else "✗"
        print(f"{status} {param_name}: {old_value} → {new_value} ({count} replacement{'s' if count != 1 else ''})")
        
        if success:
            updates_applied.append((param_name, old_value, new_value, count))
    
    print("-" * 70)
    print()
    
    # Verify updates
    print("STEP 4: Verifying changes...")
    all_good, issues = verify_updates(updated_content, updates_applied)
    
    if all_good:
        print("✓ All verifications passed!")
        print()
        
        # Write updated file
        print("STEP 5: Writing updated file...")
        write_file(filepath, updated_content)
        print(f"✓ File updated: {filepath}")
        print()
        
        # Summary
        print("=" * 70)
        print("SUCCESS! Parameters updated successfully")
        print("=" * 70)
        print()
        print(f"Updated parameters: {len(updates_applied)}")
        print(f"Backup saved as: {backup_path}")
        print()
        print("NEXT STEPS:")
        print("  1. Run a backtest to verify results")
        print("  2. Check for 7-10 trades/month")
        print("  3. Verify win rate stays >48%")
        print()
        print("To rollback:")
        print(f"  cp {backup_path} {filepath}")
        print()
        
        return 0
    
    else:
        print("✗ Verification FAILED!")
        print()
        print("Issues found:")
        for issue in issues:
            print(issue)
        print()
        print("File was NOT updated.")
        print("Please review the issues and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
