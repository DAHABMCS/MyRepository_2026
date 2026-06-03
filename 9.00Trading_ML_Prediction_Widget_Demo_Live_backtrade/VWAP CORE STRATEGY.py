# vwap_strategy_v5.py
# VWAP Scalper v5.0  |  Professional Upgrade
# Implements: Regime Detection · Pullback Logic · 3-Stage Exits · Session Filter · Mandatory/Optional Architecture
# Run: python vwap_strategy_v5.py

import tkinter as tk
from tkinter import ttk
import random
import math
import threading

# ─────────────────────────────────────────────
# STRATEGY CONFIG  v5
# ─────────────────────────────────────────────
STRATEGY_CONFIG = {
    # Capital & leverage
    "initial_capital":        50_000,
    "max_leverage":           8,
    # Base risk per setup (fraction of equity)
    "risk_reclaim":           0.004,
    "risk_bounce":            0.005,
    "risk_discount":          0.003,
    "risk_pullback":          0.004,   # NEW: EMA pullback setup
    # ATR multipliers
    "atr_period":             7,
    "stop_atr_mult":          0.8,
    "trailing_atr_mult":      0.5,
    "target_atr_mult_tp1":    1.0,    # TP1 = 1R (50% partial)
    "target_atr_mult_tp2":    2.0,    # TP2 = 2R (trailing then exits)
    # EMA / trend
    "ema_fast":               9,
    "ema_slow":               21,
    "ema_trend":              50,      # NEW: macro trend filter
    "supertrend_period":      7,
    "supertrend_multiplier":  2.0,
    "trend_signal_age_min":   3,
    # Oscillators
    "rsi_period":             9,
    "rsi_moderate_min":       42,
    "rsi_moderate_max":       62,
    "rsi_oversold":           30,
    "rsi_overbought":         70,
    # Volume
    "volume_period":          20,
    "volume_min_ratio":       0.55,
    "volume_max_ratio":       3.00,
    # ADX thresholds
    "adx_min":                22,      # mandatory trend strength
    "adx_strong":             30,      # bonus quality score
    "adx_baseline":           22,
    "adx_bounce_extra":       5,
    # Regime
    "adx_slope_bars":         5,       # bars to measure ADX slope
    "atr_regime_period":      20,      # ATR MA for volatility regime
    "atr_low_ratio":          0.7,     # ATR < 70% of avg → compressed
    "atr_high_ratio":         1.8,     # ATR > 180% of avg → spike/avoid
    # Session filter (synthetic: bar index modulo a "session length")
    "session_filter_enabled": True,
    "session_bar_start":      5,       # skip first N bars of each fake session
    "session_bar_end":        85,      # last active bar of each fake session
    "session_length":         96,      # bars per synthetic session
    # Exit logic
    "partial_profit_pct":     0.50,    # TP1 takes 50%
    "partial_profit_pct2":    0.30,    # TP2 takes another 30% (runner = 20%)
    "max_hold_bars":          60,
    # Safety
    "cooldown_bars":          20,
    "max_consecutive_losses": 3,
    "poc_avoidance_pct":      0.0025,
    # Quality gate
    "min_entry_quality":      65,
}

# ── Setup / state constants ────────────────────────────────
SETUP_VWAP_RECLAIM = "VWAP_Reclaim"
SETUP_BAND_BOUNCE  = "VWAP_Band_Bounce"
SETUP_DISCOUNT     = "Discount_Pullback"
SETUP_EMA_PULLBACK = "EMA_Pullback"      # NEW

STATE_SEEKING  = "SEEKING_ENTRY"
STATE_IN_TRADE = "IN_TRADE"
STATE_COOLDOWN = "COOLDOWN"

# Regime labels
REGIME_TRENDING    = "TRENDING"
REGIME_RANGING     = "RANGING"
REGIME_COMPRESSED  = "COMPRESSED"
REGIME_SPIKE       = "SPIKE"

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
BG     = "#0a0f1a"
BG2    = "#0d1424"
BG3    = "#111827"
BORDER = "#1e2d40"
CYAN   = "#00d4ff"
GREEN  = "#22c55e"
RED    = "#ef4444"
YELLOW = "#f59e0b"
PURPLE = "#a855f7"
MUTED  = "#6b7280"
FG     = "#e2e8f0"
FG2    = "#94a3b8"
TEAL   = "#14b8a6"
ORANGE = "#fb923c"

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def calc_ema(data, period):
    result = [0.0] * len(data)
    if len(data) < period:
        return result
    k = 2 / (period + 1)
    result[period - 1] = sum(data[:period]) / period
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result

def calc_sma(data, period):
    result = [0.0] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result

def calc_rsi(closes, period):
    result = [50.0] * len(closes)
    if len(closes) < period + 1:
        return result
    ag = al = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        if ch > 0:
            ag += ch
        else:
            al += abs(ch)
    ag /= period
    al /= period
    result[period] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g = ch if ch > 0 else 0
        l = abs(ch) if ch < 0 else 0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        result[i] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return result

def calc_stochastic(highs, lows, closes, kp=14, dp=3):
    n = len(closes)
    sk = [50.0] * n
    sd = [50.0] * n
    for i in range(kp - 1, n):
        hh = max(highs[i - kp + 1:i + 1])
        ll = min(lows[i - kp + 1:i + 1])
        sk[i] = 50 if hh == ll else 100 * (closes[i] - ll) / (hh - ll)
    for i in range(kp + dp - 2, n):
        sd[i] = sum(sk[i - dp + 1:i + 1]) / dp
    return sk, sd

def calc_atr(highs, lows, closes, period):
    n = len(closes)
    tr = [0.0] * n
    result = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    if period <= n:
        result[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result

def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    adx = [0.0] * n
    pdi = [0.0] * n
    mdi = [0.0] * n
    if n < period * 2:
        return adx, pdi, mdi
    trl, pdm, mdm = [], [], []
    for i in range(n):
        if i == 0:
            trl.append(highs[i] - lows[i])
            pdm.append(0)
            mdm.append(0)
        else:
            trl.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
            up = highs[i] - highs[i - 1]
            dn = lows[i - 1] - lows[i]
            pdm.append(up if up > dn and up > 0 else 0)
            mdm.append(dn if dn > up and dn > 0 else 0)
    str_ = sum(trl[:period])
    spdm = sum(pdm[:period])
    smdm = sum(mdm[:period])
    dx = []
    for i in range(period, n):
        if i > period:
            str_ = str_ - str_ / period + trl[i]
            spdm = spdm - spdm / period + pdm[i]
            smdm = smdm - smdm / period + mdm[i]
        pdi[i] = (spdm / str_) * 100 if str_ > 0 else 0
        mdi[i] = (smdm / str_) * 100 if str_ > 0 else 0
        ds = pdi[i] + mdi[i]
        dx.append(abs(pdi[i] - mdi[i]) / ds * 100 if ds > 0 else 0)
    if len(dx) >= period:
        adx[period * 2] = sum(dx[:period]) / period
        for i in range(period * 2 + 1, n):
            di = i - period
            if di < len(dx):
                adx[i] = (adx[i - 1] * (period - 1) + dx[di]) / period
    return adx, pdi, mdi

def calc_vwap(highs, lows, closes, volumes):
    n = len(closes)
    vw = [0.0] * n
    vh = [0.0] * n
    vl = [0.0] * n
    ctpv = cv = cvar = 0.0
    for i in range(n):
        tp = (highs[i] + lows[i] + closes[i]) / 3
        ctpv += tp * volumes[i]
        cv += volumes[i]
        if cv > 0:
            vw[i] = ctpv / cv
            dev = tp - vw[i]
            cvar += dev * dev * volumes[i]
            sd = math.sqrt(cvar / cv)
            vh[i] = vw[i] + sd
            vl[i] = vw[i] - sd
        else:
            vw[i] = vh[i] = vl[i] = closes[i]
    return vw, vh, vl

def calc_supertrend(highs, lows, closes, period, mult):
    n = len(closes)
    direction = [1] * n
    st = [0.0] * n
    age = [0] * n
    atr = calc_atr(highs, lows, closes, period)
    fu = fl = 0.0
    for i in range(n):
        hl2 = (highs[i] + lows[i]) / 2
        bu = hl2 + mult * atr[i]
        bl = hl2 - mult * atr[i]
        if i == 0 or bu < fu or closes[i - 1] > fu:
            fu = bu
        if i == 0 or bl > fl or closes[i - 1] < fl:
            fl = bl
        if i == 0:
            direction[i] = 1
            st[i] = fl
            age[i] = 1
            continue
        if direction[i - 1] == 1:
            if closes[i] < fl:
                direction[i] = -1
                st[i] = fu
            else:
                direction[i] = 1
                st[i] = fl
        else:
            if closes[i] > fu:
                direction[i] = 1
                st[i] = fl
            else:
                direction[i] = -1
                st[i] = fu
        age[i] = age[i - 1] + 1 if direction[i] == direction[i - 1] else 1
    return direction, st, age

def detect_regime(adx_vals, atr_vals, config, idx):
    """NEW: Detect market regime at bar idx."""
    if idx < config["adx_slope_bars"] + config["atr_regime_period"]:
        return REGIME_RANGING, 0.0, 1.0

    # ADX slope: rising = trending, falling = ranging
    slope_bars = config["adx_slope_bars"]
    adx_now  = adx_vals[idx]
    adx_prev = adx_vals[max(0, idx - slope_bars)]
    adx_slope = adx_now - adx_prev   # positive = trending up

    # ATR ratio: current ATR vs recent average
    atr_now = atr_vals[idx]
    atr_avg = sum(atr_vals[max(0, idx - config["atr_regime_period"]):idx]) / config["atr_regime_period"]
    atr_ratio = (atr_now / atr_avg) if atr_avg > 0 else 1.0

    if atr_ratio > config["atr_high_ratio"]:
        return REGIME_SPIKE, adx_slope, atr_ratio
    if atr_ratio < config["atr_low_ratio"]:
        return REGIME_COMPRESSED, adx_slope, atr_ratio
    if adx_now >= config["adx_min"] and adx_slope > 0:
        return REGIME_TRENDING, adx_slope, atr_ratio
    return REGIME_RANGING, adx_slope, atr_ratio

def calculate_all_indicators(candles, config):
    if len(candles) < 60:
        return [dict(c) for c in candles]
    H  = [c["high"]   for c in candles]
    L  = [c["low"]    for c in candles]
    C  = [c["close"]  for c in candles]
    V  = [c["volume"] for c in candles]

    vw, vh, vl = calc_vwap(H, L, C, V)
    e9  = calc_ema(C, config["ema_fast"])
    e21 = calc_ema(C, config["ema_slow"])
    e50 = calc_ema(C, config["ema_trend"])  # NEW
    rsi = calc_rsi(C, config["rsi_period"])
    sk, sd_ = calc_stochastic(H, L, C, 14, 3)
    atr = calc_atr(H, L, C, config["atr_period"])
    atr_slow = calc_sma(atr, config["atr_regime_period"])   # NEW
    adx, pdi, mdi = calc_adx(H, L, C, 14)
    vma = calc_sma(V, config["volume_period"])
    dr, stl, ta = calc_supertrend(H, L, C, config["supertrend_period"], config["supertrend_multiplier"])
    poc = calc_sma(C, 20)

    result = []
    for i, c in enumerate(candles):
        b = dict(c)
        atr_now = atr[i]
        atr_avg = atr_slow[i] if atr_slow[i] > 0 else atr_now
        atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

        adx_slope_bars = config["adx_slope_bars"]
        adx_slope = (adx[i] - adx[max(0, i - adx_slope_bars)]) if i >= adx_slope_bars else 0.0

        # Regime
        if atr_ratio > config["atr_high_ratio"]:
            regime = REGIME_SPIKE
        elif atr_ratio < config["atr_low_ratio"]:
            regime = REGIME_COMPRESSED
        elif adx[i] >= config["adx_min"] and adx_slope > 0:
            regime = REGIME_TRENDING
        else:
            regime = REGIME_RANGING

        b.update({
            "vwap": vw[i], "vwapHigh": vh[i], "vwapLow": vl[i],
            "ema9": e9[i], "ema21": e21[i], "ema50": e50[i],
            "rsi": rsi[i], "stochK": sk[i], "stochD": sd_[i],
            "atr": atr[i], "atrRatio": atr_ratio, "adxSlope": adx_slope,
            "adx": adx[i], "plusDI": pdi[i], "minusDI": mdi[i],
            "volumeRatio": V[i] / vma[i] if vma[i] > 0 else 0.5,
            "supertrendDir": dr[i], "trendAge": ta[i], "poc": poc[i],
            "regime": regime,
        })
        result.append(b)
    return result

def generate_synthetic_data(n=600, start=2000.0):
    candles = []
    price = start
    anchor = start
    for i in range(n):
        rev  = (anchor - price) * 0.015
        bias = random.gauss(0, 0.001)
        ret  = rev / price + bias + random.gauss(0, 0.008)
        close = round(price * (1 + ret), 2)
        rng   = close * (0.002 + random.random() * 0.006)
        high  = round(max(price, close) + rng * random.random(), 2)
        low   = round(min(price, close) - rng * random.random(), 2)
        candles.append({
            "time": i, "open": round(price, 2),
            "high": high, "low": low, "close": close,
            "volume": 5000 + random.random() * 45000
        })
        price = close
        if i % 50 == 0:
            anchor = close
    return candles

# ─────────────────────────────────────────────
# STRATEGY ENGINE  v5
# ─────────────────────────────────────────────
class VWAPStrategyEngine:
    def __init__(self, config=None):
        self.config = {**STRATEGY_CONFIG, **(config or {})}
        self._reset()

    def _reset(self):
        self.state              = STATE_SEEKING
        self.balance            = self.config["initial_capital"]
        self.position           = self._ep()
        self.trades             = []
        self.consecutive_losses = 0
        self.cooldown_counter   = 0

    def _ep(self):
        return {
            "type": None, "entryPrice": 0, "quantity": 0,
            "quantityRemaining": 0, "stopLoss": 0,
            "trailingStop": 0, "targetTP1": 0, "targetTP2": 0,
            "entryBar": 0, "barsHeld": 0, "setupType": None,
            "partial1Taken": False, "partial2Taken": False,
        }

    # ── REGIME GATE ───────────────────────────────────────────
    def _regime_ok(self, bar):
        """Only trade in TRENDING or RANGING (controlled). Block SPIKE & COMPRESSED."""
        r = bar.get("regime", REGIME_RANGING)
        if r == REGIME_SPIKE:
            return False, "Regime:SPIKE — avoid"
        if r == REGIME_COMPRESSED:
            return False, "Regime:COMPRESSED — avoid"
        return True, r

    # ── SESSION GATE ──────────────────────────────────────────
    def _session_ok(self, bar_idx):
        if not self.config["session_filter_enabled"]:
            return True
        pos = bar_idx % self.config["session_length"]
        return self.config["session_bar_start"] <= pos <= self.config["session_bar_end"]

    # ── MANDATORY TREND FILTER ────────────────────────────────
    def trend_filter(self, bar):
        sd  = bar.get("supertrendDir", 1)
        sa  = bar.get("trendAge", 0)
        e9  = bar.get("ema9", 0)
        e21 = bar.get("ema21", 0)
        e50 = bar.get("ema50", 0)
        adx = bar.get("adx", 0)

        # Mandatory gates
        if sa < self.config["trend_signal_age_min"]:
            return False, False, f"Age<{self.config['trend_signal_age_min']}({sa})"
        if adx < self.config["adx_min"]:
            return False, False, f"ADX<{self.config['adx_min']}({adx:.1f})"

        # EMA alignment (fast > slow > trend for bull; reversed for bear)
        bull_ema  = e9 > e21
        bear_ema  = e9 < e21
        # macro trend from EMA50
        bull_macro = bar.get("close", e50) > e50 * 0.999
        bear_macro = bar.get("close", e50) < e50 * 1.001

        bull = sd == 1 and bull_ema and bull_macro
        bear = sd == -1 and bear_ema and bear_macro

        if bull:
            return True, False, f"Bull[ADX:{adx:.0f} slope:{bar.get('adxSlope',0):+.1f}]"
        if bear:
            return False, True, f"Bear[ADX:{adx:.0f} slope:{bar.get('adxSlope',0):+.1f}]"
        return False, False, "EMA/ST conflict"

    # ── ENTRY SETUPS ──────────────────────────────────────────

    # 1. VWAP Reclaim (breakout through VWAP)
    def _reclaim(self, bar, pb, bull, bear):
        p   = bar["close"]
        vw  = pb.get("vwap", p)
        pc  = pb["close"]
        rsi = pb.get("rsi", 50)
        vr  = pb.get("volumeRatio", 0.5)
        poc = pb.get("poc", p)

        # Mandatory: volume and not near POC
        if vr < self.config["volume_min_ratio"]:
            return None
        if abs(p - poc) / p < self.config["poc_avoidance_pct"]:
            return None

        if bull and p > vw and pc < vw:
            q = self._quality_reclaim(pb, "long")
            if q >= self.config["min_entry_quality"]:
                return {"action": "buy", "setup": SETUP_VWAP_RECLAIM,
                        "reason": f"Reclaim↑ Q:{q}", "quality": q}
        if bear and p < vw and pc > vw:
            q = self._quality_reclaim(pb, "short")
            if q >= self.config["min_entry_quality"]:
                return {"action": "sell_short", "setup": SETUP_VWAP_RECLAIM,
                        "reason": f"Reclaim↓ Q:{q}", "quality": q}
        return None

    def _quality_reclaim(self, bar, d):
        s  = 55
        sk = bar.get("stochK", 50)
        sd = bar.get("stochD", 50)
        rsi = bar.get("rsi", 50)
        vr  = bar.get("volumeRatio", 0.5)
        adx = bar.get("adx", 0)
        if adx >= self.config["adx_strong"]:
            s += 10
        elif adx >= self.config["adx_min"]:
            s += 5
        if d == "long":
            if sk > sd and sk < 80:
                s += 10
            if 40 <= rsi <= 60:
                s += 10
            elif 30 <= rsi <= 70:
                s += 5
        else:
            if sk < sd and sk > 20:
                s += 10
            if 40 <= rsi <= 60:
                s += 10
            elif 30 <= rsi <= 70:
                s += 5
        if vr >= 0.75:
            s += 10
        elif vr >= 0.60:
            s += 5
        return min(100, s)

    # 2. VWAP Band Bounce
    def _bounce(self, bar, pb, bull, bear):
        p   = bar["close"]
        vl  = pb.get("vwapLow", p)
        vh  = pb.get("vwapHigh", p)
        pc  = pb["close"]
        adx = pb.get("adx", 0)
        pdi = pb.get("plusDI", 25)
        mdi = pb.get("minusDI", 25)
        sk  = pb.get("stochK", 50)
        sd  = pb.get("stochD", 50)
        vr  = pb.get("volumeRatio", 0.5)
        poc = pb.get("poc", p)

        if adx < self.config["adx_baseline"] + self.config["adx_bounce_extra"]:
            return None
        if vr < self.config["volume_min_ratio"]:
            return None
        if abs(p - poc) / p < self.config["poc_avoidance_pct"]:
            return None

        cross = abs(sk - sd) < 5
        if bull and p > vl and pc <= vl and pdi > mdi:
            if cross or sk > sd:
                q = self._quality_bounce(pb, "long")
                return {"action": "buy", "setup": SETUP_BAND_BOUNCE,
                        "reason": f"Bounce↑ Q:{q}", "quality": q}
        if bear and p < vh and pc >= vh and mdi > pdi:
            if cross or sk < sd:
                q = self._quality_bounce(pb, "short")
                return {"action": "sell_short", "setup": SETUP_BAND_BOUNCE,
                        "reason": f"Bounce↓ Q:{q}", "quality": q}
        return None

    def _quality_bounce(self, bar, d):
        s   = 60
        adx = bar.get("adx", 0)
        vr  = bar.get("volumeRatio", 0.5)
        pdi = bar.get("plusDI", 25)
        mdi = bar.get("minusDI", 25)
        if adx >= self.config["adx_strong"]:
            s += 12
        elif adx >= self.config["adx_min"]:
            s += 6
        if vr >= 0.75:
            s += 10
        elif vr >= 0.60:
            s += 5
        dd = (pdi - mdi) if d == "long" else (mdi - pdi)
        if dd > 10:
            s += 10
        return min(100, s)

    # 3. Discount Pullback (between VWAP low and VWAP, RSI extreme)
    def _discount(self, bar, pb, bull, bear):
        p   = bar["close"]
        vw  = pb.get("vwap", p)
        vl  = pb.get("vwapLow", p)
        vh  = pb.get("vwapHigh", p)
        rsi = pb.get("rsi", 50)
        pr  = pb.get("_prevRSI", rsi)
        vr  = pb.get("volumeRatio", 0.5)
        if vr < 0.50:
            return None
        if bull and vl <= p <= vw:
            if rsi < self.config["rsi_oversold"] or (rsi <= 40 and rsi > pr):
                q = self._quality_discount(bar, pb, "long")
                return {"action": "buy", "setup": SETUP_DISCOUNT,
                        "reason": f"Discount↑ Q:{q}", "quality": q}
        if bear and vw <= p <= vh:
            if rsi > self.config["rsi_overbought"] or (rsi >= 60 and rsi < pr):
                q = self._quality_discount(bar, pb, "short")
                return {"action": "sell_short", "setup": SETUP_DISCOUNT,
                        "reason": f"Discount↓ Q:{q}", "quality": q}
        return None

    def _quality_discount(self, bar, pb, d):
        s   = 50
        rsi = pb.get("rsi", 50)
        vw  = pb.get("vwap", 1)
        adx = pb.get("adx", 0)
        if adx >= self.config["adx_strong"]:
            s += 8
        elif adx >= self.config["adx_min"]:
            s += 4
        if d == "long":
            if rsi < 25:
                s += 15
            elif rsi < 30:
                s += 10
        else:
            if rsi > 75:
                s += 15
            elif rsi > 70:
                s += 10
        depth = abs(bar["close"] - vw) / vw * 100
        if depth > 0.5:
            s += 10
        return min(100, s)

    # 4. NEW: EMA Pullback (pullback to EMA9/21 then continuation)
    def _ema_pullback(self, bar, pb, pb2, bull, bear):
        """
        After a breakout (or confirmed trend), price pulls back to EMA9/21
        then shows resumption candle.  This is the pullback-retest-continuation
        entry that professionals prefer over pure breakouts.
        """
        p    = bar["close"]
        e9   = pb.get("ema9", p)
        e21  = pb.get("ema21", p)
        vw   = pb.get("vwap", p)
        rsi  = pb.get("rsi", 50)
        vr   = pb.get("volumeRatio", 0.5)
        low2 = pb2.get("low", p)
        high2 = pb2.get("high", p)
        adx  = pb.get("adx", 0)

        if adx < self.config["adx_min"]:
            return None
        if vr < self.config["volume_min_ratio"]:
            return None

        # Bull pullback: prev bar touched EMA9/21 zone, current bar closes above
        ema_zone_lo = min(e9, e21)
        ema_zone_hi = max(e9, e21)

        if bull:
            touched_ema = low2 <= ema_zone_hi * 1.002  # within 0.2% of EMA
            resuming    = p > e9 and p > vw * 0.999
            rsi_ok      = rsi >= 40  # not oversold breakdown
            if touched_ema and resuming and rsi_ok:
                q = self._quality_ema_pullback(pb, "long")
                if q >= self.config["min_entry_quality"]:
                    return {"action": "buy", "setup": SETUP_EMA_PULLBACK,
                            "reason": f"EMA-PB↑ Q:{q}", "quality": q}

        if bear:
            touched_ema = high2 >= ema_zone_lo * 0.998
            resuming    = p < e9 and p < vw * 1.001
            rsi_ok      = rsi <= 60
            if touched_ema and resuming and rsi_ok:
                q = self._quality_ema_pullback(pb, "short")
                if q >= self.config["min_entry_quality"]:
                    return {"action": "sell_short", "setup": SETUP_EMA_PULLBACK,
                            "reason": f"EMA-PB↓ Q:{q}", "quality": q}
        return None

    def _quality_ema_pullback(self, bar, d):
        s   = 60
        adx = bar.get("adx", 0)
        vr  = bar.get("volumeRatio", 0.5)
        atr_ratio = bar.get("atrRatio", 1.0)
        adx_slope = bar.get("adxSlope", 0)
        if adx >= self.config["adx_strong"]:
            s += 12
        elif adx >= self.config["adx_min"]:
            s += 6
        if vr >= 0.8:
            s += 10
        elif vr >= 0.65:
            s += 5
        if adx_slope > 1:       # ADX rising = trend accelerating
            s += 8
        if 0.9 <= atr_ratio <= 1.5:   # healthy volatility
            s += 5
        return min(100, s)

    # ── POSITION SIZING ───────────────────────────────────────
    def _pos_size(self, price, atr, setup, bar):
        cfg = self.config
        risk_map = {
            SETUP_VWAP_RECLAIM: cfg["risk_reclaim"],
            SETUP_BAND_BOUNCE:  cfg["risk_bounce"],
            SETUP_DISCOUNT:     cfg["risk_discount"],
            SETUP_EMA_PULLBACK: cfg["risk_pullback"],
        }
        rp = risk_map.get(setup, cfg["risk_reclaim"])

        # Regime-based position scaling
        regime = bar.get("regime", REGIME_RANGING)
        atr_ratio = bar.get("atrRatio", 1.0)
        if regime == REGIME_TRENDING:
            rp *= 1.1    # slight boost in strong trend
        elif regime == REGIME_RANGING:
            rp *= 0.85   # reduce in ranging
        if atr_ratio < 0.85:
            rp *= 0.80   # reduce in compressed vol

        # Dynamic scaling from recent performance
        recent = self.trades[-10:]
        if len(recent) >= 5:
            wr = sum(1 for t in recent if t["pnl"] > 0) / len(recent)
            if wr > 0.60:
                rp *= 1.10
            elif wr < 0.45:
                rp *= 0.80

        ra   = self.balance * rp
        stop_dist = (atr or price * 0.01) * cfg["stop_atr_mult"]
        size = ra / max(stop_dist, price * 0.001)
        max_size = (self.balance * cfg["max_leverage"]) / price
        return max(min(size, max_size), 0.001)

    # ── ENTRY ORCHESTRATOR ────────────────────────────────────
    def check_entry(self, bar, pb, pb2, settings, bar_idx):
        if self.state == STATE_COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.config["cooldown_bars"]:
                self.state = STATE_SEEKING
                self.cooldown_counter = 0
                self.consecutive_losses = 0
            else:
                return {"action": "hold", "reason": f"Cooldown({self.cooldown_counter}/{self.config['cooldown_bars']})"}

        if self.state == STATE_IN_TRADE:
            return {"action": "hold", "reason": "In trade"}

        # Session gate
        if not self._session_ok(bar_idx):
            return {"action": "hold", "reason": "Off-session"}

        # Regime gate
        regime_ok, regime_msg = self._regime_ok(bar)
        if not regime_ok:
            return {"action": "hold", "reason": regime_msg}

        # Mandatory trend
        bull, bear, tr = self.trend_filter(pb)
        if not bull and not bear:
            return {"action": "hold", "reason": tr}

        ap = {**pb, "_prevRSI": pb2.get("rsi", pb.get("rsi", 50))}

        # Check each enabled setup (mandatory filters already passed)
        if settings.get("enableReclaim", True):
            s = self._reclaim(bar, ap, bull, bear)
            if s:
                return s

        if settings.get("enableBounce", True):
            s = self._bounce(bar, ap, bull, bear)
            if s:
                return s

        if settings.get("enableDiscount", True):
            s = self._discount(bar, ap, bull, bear)
            if s:
                return s

        if settings.get("enablePullback", True):
            s = self._ema_pullback(bar, ap, pb2, bull, bear)
            if s:
                return s

        return {"action": "hold", "reason": f"No setup ({tr})"}

    # ── EXIT LOGIC  (3-stage: TP1 50% · TP2 30% trailing · runner until ST flip) ──
    def check_exit(self, bar, bi):
        if self.state != STATE_IN_TRADE:
            return None
        price = bar["close"]
        pos   = self.position
        atr   = bar.get("atr", 0.01)
        il    = pos["type"] == "long"

        # Hard stop
        if (il and price <= pos["stopLoss"]) or (not il and price >= pos["stopLoss"]):
            return {"action": "exit", "reason": "Stop loss"}

        # TP1: 50% at 1R
        if not pos["partial1Taken"]:
            hit = price >= pos["targetTP1"] if il else price <= pos["targetTP1"]
            if hit:
                return {"action": "partial1", "reason": "TP1 (1R, 50%)"}

        # TP2: 30% at 2R
        if pos["partial1Taken"] and not pos["partial2Taken"]:
            hit2 = price >= pos["targetTP2"] if il else price <= pos["targetTP2"]
            if hit2:
                return {"action": "partial2", "reason": "TP2 (2R, 30%)"}

        # Trailing stop on runner (post TP1)
        if pos["partial1Taken"] and pos["trailingStop"] > 0:
            if (il and price <= pos["trailingStop"]) or (not il and price >= pos["trailingStop"]):
                return {"action": "exit", "reason": "Trailing stop"}
            td = atr * self.config["trailing_atr_mult"]
            if il:
                nt = price - td
                if nt > pos["trailingStop"]:
                    pos["trailingStop"] = nt
            else:
                nt = price + td
                if nt < pos["trailingStop"]:
                    pos["trailingStop"] = nt

        # SuperTrend flip exits runner
        if pos["partial1Taken"]:
            if (il and bar.get("supertrendDir") == -1) or (not il and bar.get("supertrendDir") == 1):
                return {"action": "exit", "reason": "SuperTrend flip"}

        # VWAP loss exits runner
        vw = bar.get("vwap", price)
        if pos["partial1Taken"]:
            if (il and price < vw * 0.9985) or (not il and price > vw * 1.0015):
                return {"action": "exit", "reason": "VWAP loss (runner)"}

        # Time stop
        pos["barsHeld"] += 1
        if pos["barsHeld"] >= self.config["max_hold_bars"]:
            return {"action": "exit", "reason": f"Max hold ({self.config['max_hold_bars']}b)"}
        return None

    # ── EXECUTION ─────────────────────────────────────────────
    def exec_entry(self, action, price, atr, setup, bi, bar):
        qty = self._pos_size(price, atr, setup, bar)
        sd  = (atr or price * 0.01) * self.config["stop_atr_mult"]
        il  = action == "buy"
        tp1 = price + atr * self.config["target_atr_mult_tp1"] if il else price - atr * self.config["target_atr_mult_tp1"]
        tp2 = price + atr * self.config["target_atr_mult_tp2"] if il else price - atr * self.config["target_atr_mult_tp2"]
        self.position = {
            "type": "long" if il else "short",
            "entryPrice": price, "quantity": qty, "quantityRemaining": qty,
            "stopLoss": price - sd if il else price + sd,
            "trailingStop": 0,
            "targetTP1": tp1, "targetTP2": tp2,
            "entryBar": bi, "barsHeld": 0, "setupType": setup,
            "partial1Taken": False, "partial2Taken": False,
        }
        self.state = STATE_IN_TRADE

    def exec_exit(self, price, reason, bi, mode="full"):
        pos   = self.position
        il    = pos["type"] == "long"
        cfg   = self.config

        if mode == "partial1":
            qty  = pos["quantity"] * cfg["partial_profit_pct"]
            pnl  = (price - pos["entryPrice"]) * qty if il else (pos["entryPrice"] - price) * qty
            pos["quantityRemaining"] -= qty
            pos["quantity"]          = pos["quantityRemaining"]
            pos["partial1Taken"]     = True
            # Set trailing stop after TP1
            atr_approx = abs(pos["targetTP1"] - pos["entryPrice"])
            td = atr_approx * cfg["trailing_atr_mult"]
            pos["trailingStop"] = price - td if il else price + td
            self.balance += pnl
            return {"pnl": pnl, "mode": "partial1"}

        if mode == "partial2":
            qty  = pos["quantity"] * (cfg["partial_profit_pct2"] / (1 - cfg["partial_profit_pct"]))
            qty  = min(qty, pos["quantity"] * 0.6)  # safety cap
            pnl  = (price - pos["entryPrice"]) * qty if il else (pos["entryPrice"] - price) * qty
            pos["quantityRemaining"] -= qty
            pos["quantity"]          = pos["quantityRemaining"]
            pos["partial2Taken"]     = True
            self.balance += pnl
            return {"pnl": pnl, "mode": "partial2"}

        # Full exit (runner or stop)
        qty = pos["quantity"]
        pnl = (price - pos["entryPrice"]) * qty if il else (pos["entryPrice"] - price) * qty
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config["max_consecutive_losses"]:
                self.state = STATE_COOLDOWN
                self.cooldown_counter = 0
        else:
            self.consecutive_losses = 0

        trade = {
            "entryBar": pos["entryBar"], "exitBar": bi,
            "entryPrice": pos["entryPrice"], "exitPrice": price,
            "quantity": pos["quantity"], "type": pos["type"],
            "setupType": pos["setupType"],
            "pnl": pnl, "exitReason": reason, "barsHeld": pos["barsHeld"],
        }
        self.trades.append(trade)
        self.balance += pnl
        self.position = self._ep()
        if self.state != STATE_COOLDOWN:
            self.state = STATE_SEEKING
        return {"pnl": pnl, "mode": "full", "trade": trade}

    # ── BACKTEST RUNNER ───────────────────────────────────────
    def run_backtest(self, bars, settings=None):
        if settings is None:
            settings = {"enableReclaim": True, "enableBounce": True,
                        "enableDiscount": True, "enablePullback": True}
        self._reset()
        eq = [self.config["initial_capital"]]

        for i in range(3, len(bars)):
            bar = bars[i]
            pb  = {**bars[i - 1], "_prevRSI": bars[i - 2].get("rsi", 50)}
            pb2 = bars[i - 2]

            if self.state == STATE_IN_TRADE:
                es = self.check_exit(bar, i)
                if es:
                    if es["action"] == "partial1":
                        self.exec_exit(bar["close"], es["reason"], i, "partial1")
                    elif es["action"] == "partial2":
                        self.exec_exit(bar["close"], es["reason"], i, "partial2")
                    else:
                        self.exec_exit(bar["close"], es["reason"], i, "full")

            if self.state != STATE_IN_TRADE:
                sig = self.check_entry(bar, pb, pb2, settings, i)
                if sig["action"] in ("buy", "sell_short"):
                    self.exec_entry(sig["action"], bar["close"],
                                    bar.get("atr") or bar["close"] * 0.01,
                                    sig["setup"], i, bar)
            eq.append(self.balance)

        if self.state == STATE_IN_TRADE:
            self.exec_exit(bars[-1]["close"], "End of data", len(bars) - 1, "full")
            eq.append(self.balance)

        return {"trades": self.trades, "equity_curve": eq, "stats": self.get_stats()}

    def get_stats(self):
        if not self.trades:
            return {"totalTrades": 0, "winRate": 0, "profitFactor": 0,
                    "totalPnl": 0, "balance": self.balance, "roi": 0,
                    "avgWin": 0, "avgLoss": 0, "maxDrawdown": 0,
                    "sharpeRatio": 0, "winningTrades": 0, "losingTrades": 0,
                    "setupBreakdown": {}}
        wins   = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        tp  = sum(t["pnl"] for t in self.trades)
        tw  = sum(t["pnl"] for t in wins)
        tl  = abs(sum(t["pnl"] for t in losses))
        peak = self.config["initial_capital"]
        mdd  = 0
        run  = peak
        for t in self.trades:
            run += t["pnl"]
            if run > peak:
                peak = run
            dd = (peak - run) / peak * 100
            if dd > mdd:
                mdd = dd
        rets = [t["pnl"] / self.config["initial_capital"] for t in self.trades]
        ar   = sum(rets) / len(rets)
        var  = sum((r - ar) ** 2 for r in rets) / len(rets)
        sd   = math.sqrt(var)
        sharpe = (ar / sd) * math.sqrt(252) if sd > 0 else 0

        # Setup breakdown
        sbd = {}
        for setup in [SETUP_VWAP_RECLAIM, SETUP_BAND_BOUNCE, SETUP_DISCOUNT, SETUP_EMA_PULLBACK]:
            ts = [t for t in self.trades if t["setupType"] == setup]
            if ts:
                ws = sum(1 for t in ts if t["pnl"] > 0)
                sbd[setup] = {"count": len(ts), "winRate": ws / len(ts) * 100,
                               "pnl": sum(t["pnl"] for t in ts)}

        return {
            "totalTrades": len(self.trades), "winningTrades": len(wins),
            "losingTrades": len(losses),
            "winRate": len(wins) / len(self.trades) * 100,
            "profitFactor": tw / tl if tl > 0 else (float("inf") if tw > 0 else 0),
            "totalPnl": tp, "balance": self.balance,
            "roi": (self.balance - self.config["initial_capital"]) / self.config["initial_capital"] * 100,
            "avgWin": tw / len(wins) if wins else 0,
            "avgLoss": tl / len(losses) if losses else 0,
            "maxDrawdown": mdd, "sharpeRatio": sharpe,
            "setupBreakdown": sbd,
        }

# ─────────────────────────────────────────────
# GUI  v5
# ─────────────────────────────────────────────
class VWAPDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VWAP Scalper v5.0  |  Professional Trading Dashboard")
        self.geometry("1480x900")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._bars        = []
        self._trades      = []
        self._stats       = None
        self._equity      = []
        self._sim_running = False
        self._sim_thread  = None
        self._candles     = []
        self._bar_idx     = 0
        self._engine      = None

        self._build_ui()

    # ─────────────── BUILD UI ─────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG2, height=46)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="● VWAP Scalper", fg=CYAN, bg=BG2,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(hdr, text="v5.0  PRO", fg=PURPLE, bg=BG2,
                 font=("Courier New", 11)).pack(side="left")
        # Regime indicator (live)
        self._regime_var = tk.StringVar(value="REGIME: —")
        tk.Label(hdr, textvariable=self._regime_var, fg=YELLOW, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(side="left", padx=24)

        btn_frame = tk.Frame(hdr, bg=BG2)
        btn_frame.pack(side="right", padx=12)
        self._btn_backtest = self._btn(btn_frame, "▶ Backtest", self._run_backtest, CYAN)
        self._btn_backtest.pack(side="left", padx=3)
        self._btn_simulate = self._btn(btn_frame, "⚡ Simulate", self._start_sim, GREEN)
        self._btn_simulate.pack(side="left", padx=3)
        self._btn_stop = self._btn(btn_frame, "■ Stop", self._stop_sim, RED)
        self._btn_stop.pack(side="left", padx=3)
        self._btn_stop.config(state="disabled")
        self._btn_reset = self._btn(btn_frame, "↺ Reset", self._reset, MUTED)
        self._btn_reset.pack(side="left", padx=3)

        # Banner
        self._banner = tk.Frame(self, bg=BG3, height=56)
        self._banner.pack(fill="x")
        self._banner.pack_propagate(False)
        self._build_banner()

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG2, width=230)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_sidebar(left)

        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True)
        self._build_center(center)

        right = tk.Frame(body, bg=BG2, width=270)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_right(right)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=BG3, fg=color,
                         font=("Courier New", 9, "bold"), relief="flat",
                         padx=10, pady=4, cursor="hand2",
                         activebackground=BORDER, activeforeground=color)

    # ── BANNER ────────────────────────────────
    def _build_banner(self):
        metrics = [
            ("TOTAL TRADES", "—", MUTED), ("WIN RATE",      "—", MUTED),
            ("PROFIT FACTOR","—", MUTED), ("TOTAL PnL",     "—", MUTED),
            ("MAX DRAWDOWN", "—", MUTED), ("SHARPE",        "—", MUTED),
            ("ROI",          "—", MUTED), ("BALANCE", "$50,000.00", CYAN),
        ]
        self._banner_vars = {}
        for label, val, color in metrics:
            f = tk.Frame(self._banner, bg=BG3)
            f.pack(side="left", padx=16, pady=8)
            tk.Label(f, text=label, fg=MUTED, bg=BG3, font=("Courier New", 7)).pack(anchor="w")
            var = tk.StringVar(value=val)
            lbl = tk.Label(f, textvariable=var, fg=color, bg=BG3,
                           font=("Courier New", 11, "bold"))
            lbl.pack(anchor="w")
            self._banner_vars[label] = (var, lbl)

    def _update_banner(self, stats):
        def fmt(label, val, color):
            v, l = self._banner_vars[label]
            v.set(val); l.config(fg=color)
        fmt("TOTAL TRADES", str(stats["totalTrades"]), CYAN)
        wr = stats["winRate"]
        fmt("WIN RATE", f"{wr:.1f}%", GREEN if wr >= 50 else RED)
        pf = stats["profitFactor"]
        fmt("PROFIT FACTOR", "∞" if pf == float("inf") else f"{pf:.2f}",
            GREEN if pf >= 1.5 else (YELLOW if pf >= 1 else RED))
        pnl = stats["totalPnl"]
        fmt("TOTAL PnL", f"{'+'if pnl>=0 else ''}${pnl:,.2f}", GREEN if pnl >= 0 else RED)
        dd = stats["maxDrawdown"]
        fmt("MAX DRAWDOWN", f"{dd:.2f}%", GREEN if dd < 5 else (YELLOW if dd < 10 else RED))
        sh = stats["sharpeRatio"]
        fmt("SHARPE", f"{sh:.2f}", GREEN if sh >= 1 else (YELLOW if sh >= 0 else RED))
        roi = stats["roi"]
        fmt("ROI", f"{'+'if roi>=0 else ''}{roi:.2f}%", GREEN if roi >= 0 else RED)
        bal = stats["balance"]
        fmt("BALANCE", f"${bal:,.2f}",
            GREEN if bal >= STRATEGY_CONFIG["initial_capital"] else RED)

    # ── SIDEBAR ───────────────────────────────
    def _build_sidebar(self, parent):
        tk.Label(parent, text="SETTINGS", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(12, 4))

        def row(label, widget_fn):
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=3)
            tk.Label(f, text=label, fg=MUTED, bg=BG2, font=("Courier New", 8)).pack(anchor="w")
            w = widget_fn(f)
            w.pack(fill="x", pady=1)
            return w

        self._sym_var = tk.StringVar(value="ETH-USDT")
        row("SYMBOL", lambda f: ttk.Combobox(f, textvariable=self._sym_var,
            values=["ETH-USDT", "BTC-USDT", "SOL-USDT"], state="readonly",
            font=("Courier New", 9)))

        self._tf_var = tk.StringVar(value="5m")
        row("TIMEFRAME", lambda f: ttk.Combobox(f, textvariable=self._tf_var,
            values=["1m", "3m", "5m", "15m", "1H"], state="readonly",
            font=("Courier New", 9)))

        self._dir_var = tk.StringVar(value="Both")
        row("DIRECTION", lambda f: ttk.Combobox(f, textvariable=self._dir_var,
            values=["Both", "Long Only", "Short Only"], state="readonly",
            font=("Courier New", 9)))

        # Setups (4 now)
        tk.Label(parent, text="SETUPS", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(14, 4))
        self._rec_var = tk.BooleanVar(value=True)
        self._bou_var = tk.BooleanVar(value=True)
        self._dis_var = tk.BooleanVar(value=True)
        self._pb_var  = tk.BooleanVar(value=True)

        for var, label, sub in [
            (self._rec_var, "VWAP Reclaim",   "61-63% WR"),
            (self._bou_var, "Band Bounce",    "62-65% WR"),
            (self._dis_var, "Discount PB",    "57-60% WR"),
            (self._pb_var,  "EMA Pullback ★", "63-66% WR"),
        ]:
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=2)
            tk.Checkbutton(f, text=label, variable=var, bg=BG2, fg=FG,
                           selectcolor=BG3, activebackground=BG2, activeforeground=CYAN,
                           font=("Courier New", 8)).pack(side="left")
            tk.Label(f, text=sub, fg=MUTED, bg=BG2, font=("Courier New", 7)).pack(side="right")

        # Regime filter toggle
        tk.Label(parent, text="FILTERS", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(14, 4))
        self._session_var = tk.BooleanVar(value=True)
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(f, text="Session Filter", variable=self._session_var,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       activeforeground=CYAN, font=("Courier New", 8)).pack(side="left")
        self._regime_filter_var = tk.BooleanVar(value=True)
        f2 = tk.Frame(parent, bg=BG2)
        f2.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(f2, text="Regime Filter", variable=self._regime_filter_var,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       activeforeground=CYAN, font=("Courier New", 8)).pack(side="left")

        # Risk params
        tk.Label(parent, text="RISK PARAMS", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(14, 4))
        for label, val in [
            ("Stop ATR ×",   "0.8×"),
            ("TP1 ATR ×",    "1.0×  (50%)"),
            ("TP2 ATR ×",    "2.0×  (30%)"),
            ("Runner",       "20%  trail+ST"),
            ("Max Leverage", "8×"),
            ("Capital",      "$50,000"),
        ]:
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=1)
            tk.Label(f, text=label, fg=MUTED, bg=BG2, font=("Courier New", 7)).pack(side="left")
            tk.Label(f, text=val, fg=FG2, bg=BG2,
                     font=("Courier New", 8, "bold")).pack(side="right")

    # ── CENTER ────────────────────────────────
    def _build_center(self, parent):
        # Stats row
        stats_row = tk.Frame(parent, bg=BG3, height=80)
        stats_row.pack(fill="x")
        stats_row.pack_propagate(False)
        self._stat_frames = {}
        for label, val, color in [
            ("W Trades", "—", GREEN), ("L Trades", "—", RED),
            ("Avg Win",  "—", GREEN), ("Avg Loss", "—", RED),
            ("Bars",     "—", MUTED), ("Drawdown", "—", YELLOW),
        ]:
            f = tk.Frame(stats_row, bg=BG3)
            f.pack(side="left", padx=20, pady=12)
            tk.Label(f, text=label, fg=MUTED, bg=BG3, font=("Courier New", 7)).pack(anchor="w")
            var = tk.StringVar(value=val)
            lbl = tk.Label(f, textvariable=var, fg=color, bg=BG3,
                           font=("Courier New", 12, "bold"))
            lbl.pack(anchor="w")
            self._stat_frames[label] = (var, lbl)

        # Tabs
        tab_bar = tk.Frame(parent, bg=BG2, height=30)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)
        self._tab_var = tk.StringVar(value="trades")
        for txt, val in [("📈 Equity", "equity"),
                         ("📋 Trade Log", "trades"),
                         ("📊 Setup Stats", "setup")]:
            tk.Radiobutton(tab_bar, text=txt, variable=self._tab_var, value=val,
                           bg=BG2, fg=FG2, selectcolor=BG3, activebackground=BG2,
                           font=("Courier New", 8), command=self._switch_tab,
                           indicatoron=False, relief="flat", padx=10).pack(side="left")

        self._tab_content = tk.Frame(parent, bg=BG)
        self._tab_content.pack(fill="both", expand=True)
        self._equity_frame = tk.Frame(self._tab_content, bg=BG)
        self._trade_frame  = tk.Frame(self._tab_content, bg=BG)
        self._setup_frame  = tk.Frame(self._tab_content, bg=BG)  # NEW
        self._build_equity_canvas(self._equity_frame)
        self._build_trade_table(self._trade_frame)
        self._build_setup_panel(self._setup_frame)
        self._switch_tab()

    def _switch_tab(self):
        val = self._tab_var.get()
        self._equity_frame.pack_forget()
        self._trade_frame.pack_forget()
        self._setup_frame.pack_forget()
        if val == "equity":
            self._equity_frame.pack(fill="both", expand=True)
        elif val == "trades":
            self._trade_frame.pack(fill="both", expand=True)
        else:
            self._setup_frame.pack(fill="both", expand=True)

    # ── EQUITY CANVAS ─────────────────────────
    def _build_equity_canvas(self, parent):
        self._eq_canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        self._eq_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self._eq_canvas.bind("<Configure>", lambda e: self._draw_equity())

    def _draw_equity(self):
        c = self._eq_canvas
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10 or not self._equity:
            c.create_text(W // 2, H // 2, text="Run a backtest to see equity curve",
                          fill=MUTED, font=("Courier New", 10))
            return
        data = self._equity
        mn = min(data); mx = max(data)
        rng = mx - mn or 1
        pad = 50
        pts = []
        for i, v in enumerate(data):
            x = pad + (i / max(len(data) - 1, 1)) * (W - pad * 2)
            y = H - pad - ((v - mn) / rng) * (H - pad * 2)
            pts.extend([x, y])

        color = GREEN if data[-1] >= data[0] else RED
        if len(pts) >= 4:
            baseline = pts[:] + [pts[-2], H - pad, pad, H - pad]
            c.create_polygon(baseline, fill=color, stipple="gray12", outline="")
            c.create_line(pts, fill=color, width=2, smooth=True)

        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            y = H - pad - frac * (H - pad * 2)
            val = mn + frac * rng
            c.create_line(pad, y, W - pad, y, fill=BORDER, dash=(2, 4))
            c.create_text(pad - 4, y, text=f"${val:,.0f}", fill=MUTED,
                          font=("Courier New", 7), anchor="e")

        c.create_text(pad, pad - 14, text=f"${mx:,.0f}", fill=MUTED,
                      font=("Courier New", 7), anchor="w")
        c.create_text(pad, H - pad + 10, text=f"${mn:,.0f}", fill=MUTED,
                      font=("Courier New", 7), anchor="w")
        c.create_text(W // 2, 10, text="Equity Curve", fill=FG2,
                      font=("Courier New", 9, "bold"))

    # ── TRADE TABLE ───────────────────────────
    def _build_trade_table(self, parent):
        self._trade_summary = tk.Label(parent, text="No trades yet", fg=MUTED, bg=BG3,
                                       font=("Courier New", 8), anchor="w", padx=10, pady=4)
        self._trade_summary.pack(fill="x")
        cols = ("#", "Dir", "Setup", "Entry $", "Exit $", "Bars", "Exit Reason", "P&L")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=22)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=BG, foreground=FG,
                        fieldbackground=BG, font=("Courier New", 8), rowheight=22)
        style.configure("Treeview.Heading", background=BG3, foreground=CYAN,
                        font=("Courier New", 8, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", BORDER)])
        widths = [35, 65, 120, 85, 85, 45, 160, 100]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w,
                              anchor="center" if col in ("#", "Bars") else "w")
        self._tree.tag_configure("win",  foreground=GREEN)
        self._tree.tag_configure("loss", foreground=RED)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _populate_trade_table(self, trades):
        self._tree.delete(*self._tree.get_children())
        if not trades:
            self._trade_summary.config(text="No trades yet", fg=MUTED)
            return
        wins = sum(1 for t in trades if t["pnl"] > 0)
        net  = sum(t["pnl"] for t in trades)
        self._trade_summary.config(
            text=f"  {len(trades)} trades  |  {wins}W / {len(trades)-wins}L  |  "
                 f"WR: {wins/len(trades)*100:.1f}%  |  Net: {'+'if net>=0 else ''}${net:,.2f}",
            fg=GREEN if net >= 0 else RED)
        for i, t in enumerate(trades, 1):
            direction = "▲ LONG" if t["type"] == "long" else "▼ SHORT"
            setup = (t["setupType"] or "—").replace("VWAP_", "").replace("_", " ")
            pct   = abs((t["exitPrice"] - t["entryPrice"]) / t["entryPrice"] * 100)
            pnl_s = f"{'+'if t['pnl']>=0 else ''}${t['pnl']:,.2f} ({pct:.2f}%)"
            tag   = "win" if t["pnl"] >= 0 else "loss"
            self._tree.insert("", "end", values=(
                i, direction, setup,
                f"${t['entryPrice']:.2f}", f"${t['exitPrice']:.2f}",
                t["barsHeld"], t["exitReason"], pnl_s
            ), tags=(tag,))

    # ── SETUP BREAKDOWN PANEL (NEW) ──────────
    def _build_setup_panel(self, parent):
        tk.Label(parent, text="SETUP PERFORMANCE BREAKDOWN", fg=CYAN, bg=BG,
                 font=("Courier New", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        self._setup_labels = {}
        colors = {
            SETUP_VWAP_RECLAIM: CYAN,
            SETUP_BAND_BOUNCE:  TEAL,
            SETUP_DISCOUNT:     YELLOW,
            SETUP_EMA_PULLBACK: PURPLE,
        }
        for setup in [SETUP_VWAP_RECLAIM, SETUP_BAND_BOUNCE, SETUP_DISCOUNT, SETUP_EMA_PULLBACK]:
            f = tk.Frame(parent, bg=BG3)
            f.pack(fill="x", padx=14, pady=4)
            short = setup.replace("VWAP_", "").replace("_", " ")
            color = colors[setup]
            tk.Label(f, text=f"  {short}", fg=color, bg=BG3,
                     font=("Courier New", 9, "bold"), width=18, anchor="w").pack(side="left", padx=6, pady=6)
            for key in ["Trades", "Win%", "Net PnL"]:
                sf = tk.Frame(f, bg=BG3)
                sf.pack(side="left", padx=10)
                tk.Label(sf, text=key, fg=MUTED, bg=BG3, font=("Courier New", 7)).pack(anchor="w")
                var = tk.StringVar(value="—")
                tk.Label(sf, textvariable=var, fg=FG2, bg=BG3,
                         font=("Courier New", 10, "bold")).pack(anchor="w")
                self._setup_labels[(setup, key)] = var

    def _update_setup_panel(self, sbd):
        for setup in [SETUP_VWAP_RECLAIM, SETUP_BAND_BOUNCE, SETUP_DISCOUNT, SETUP_EMA_PULLBACK]:
            if setup in sbd:
                d = sbd[setup]
                self._setup_labels[(setup, "Trades")].set(str(d["count"]))
                self._setup_labels[(setup, "Win%")].set(f"{d['winRate']:.1f}%")
                pnl = d["pnl"]
                self._setup_labels[(setup, "Net PnL")].set(
                    f"{'+'if pnl>=0 else ''}${pnl:,.0f}")
            else:
                for key in ["Trades", "Win%", "Net PnL"]:
                    self._setup_labels[(setup, key)].set("—")

    # ── RIGHT PANEL ───────────────────────────
    def _build_right(self, parent):
        tk.Label(parent, text="POSITION", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(12, 4))
        self._pos_frame  = tk.Frame(parent, bg=BG2)
        self._pos_frame.pack(fill="x", padx=8)
        self._pos_labels = {}
        for key in ["Status", "Type", "Entry", "Stop", "TP1", "TP2", "Qty", "Bars", "P1", "P2"]:
            f = tk.Frame(self._pos_frame, bg=BG2)
            f.pack(fill="x", pady=1)
            tk.Label(f, text=key, fg=MUTED, bg=BG2, font=("Courier New", 7)).pack(side="left")
            var = tk.StringVar(value="—")
            tk.Label(f, textvariable=var, fg=FG2, bg=BG2,
                     font=("Courier New", 8, "bold")).pack(side="right")
            self._pos_labels[key] = var
        self._pos_labels["Status"].set("Seeking Entry")

        tk.Label(parent, text="TRADING LOG", fg=CYAN, bg=BG2,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=10, pady=(16, 4))
        log_frame = tk.Frame(parent, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self._log_box = tk.Text(log_frame, bg=BG, fg=FG2, font=("Courier New", 7),
                                state="disabled", wrap="word", relief="flat")
        log_sb = ttk.Scrollbar(log_frame, command=self._log_box.yview)
        self._log_box.configure(yscrollcommand=log_sb.set)
        for tag, color in [
            ("entry_long",  GREEN), ("entry_short", RED),
            ("exit_win",    GREEN), ("exit_loss",   RED),
            ("partial",     YELLOW), ("warn",       YELLOW),
            ("info",        MUTED),  ("regime",     ORANGE),
        ]:
            self._log_box.tag_config(tag, foreground=color)
        self._log_box.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

    def _log(self, msg, tag="info"):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg + "\n", tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _update_stats_row(self, stats, n_bars):
        def s(key, val, color=None):
            var, lbl = self._stat_frames[key]
            var.set(val)
            if color:
                lbl.config(fg=color)
        s("W Trades", str(stats["winningTrades"]), GREEN)
        s("L Trades", str(stats["losingTrades"]), RED)
        s("Avg Win",  f"${stats['avgWin']:.2f}", GREEN)
        s("Avg Loss", f"${stats['avgLoss']:.2f}", RED)
        s("Bars",     str(n_bars), MUTED)
        dd = stats["maxDrawdown"]
        s("Drawdown", f"{dd:.2f}%",
          GREEN if dd < 5 else (YELLOW if dd < 10 else RED))

    def _update_position(self, engine):
        pos = engine.position
        if not pos["type"]:
            self._pos_labels["Status"].set(engine.state)
            for k in ["Type", "Entry", "Stop", "TP1", "TP2", "Qty", "Bars", "P1", "P2"]:
                self._pos_labels[k].set("—")
        else:
            self._pos_labels["Status"].set("IN TRADE")
            self._pos_labels["Type"].set(pos["type"].upper())
            self._pos_labels["Entry"].set(f"${pos['entryPrice']:.2f}")
            self._pos_labels["Stop"].set(f"${pos['stopLoss']:.2f}")
            self._pos_labels["TP1"].set(f"${pos['targetTP1']:.2f}")
            self._pos_labels["TP2"].set(f"${pos['targetTP2']:.2f}")
            self._pos_labels["Qty"].set(f"{pos['quantity']:.4f}")
            self._pos_labels["Bars"].set(str(pos["barsHeld"]))
            self._pos_labels["P1"].set("✓" if pos["partial1Taken"] else "—")
            self._pos_labels["P2"].set("✓" if pos["partial2Taken"] else "—")

    # ── SETTINGS GETTER ───────────────────────
    def _get_settings(self):
        STRATEGY_CONFIG["session_filter_enabled"] = self._session_var.get()
        return {
            "enableReclaim":  self._rec_var.get(),
            "enableBounce":   self._bou_var.get(),
            "enableDiscount": self._dis_var.get(),
            "enablePullback": self._pb_var.get(),
        }

    # ── BACKTEST ──────────────────────────────
    def _run_backtest(self):
        self._clear_log()
        self._log("═══ BACKTEST v5 STARTED ═══", "info")
        self._log(f"Symbol: {self._sym_var.get()} | TF: {self._tf_var.get()}", "info")
        self._log("Regime detection: ON  |  Session filter: ON  |  EMA Pullback: ON", "regime")

        candles = generate_synthetic_data(600)
        self._log(f"Generated {len(candles)} synthetic candles", "info")
        bars = calculate_all_indicators(candles, STRATEGY_CONFIG)
        self._log(f"All indicators computed ({len(bars)} bars)", "info")

        engine = VWAPStrategyEngine()
        result = engine.run_backtest(bars, self._get_settings())

        self._bars   = bars
        self._trades = result["trades"]
        self._stats  = result["stats"]
        self._equity = result["equity_curve"]

        stats = result["stats"]
        self._update_banner(stats)
        self._update_stats_row(stats, len(bars))
        self._populate_trade_table(result["trades"])
        self._update_setup_panel(stats.get("setupBreakdown", {}))
        self._draw_equity()
        self._update_position(engine)

        for t in result["trades"]:
            arrow  = "▲" if t["type"] == "long" else "▼"
            tag_e  = "entry_long" if t["type"] == "long" else "entry_short"
            self._log(f"{arrow} {t['type'].upper()} @ ${t['entryPrice']:.2f} | {t['setupType']}", tag_e)
            sign   = "+" if t["pnl"] >= 0 else ""
            tag_x  = "exit_win" if t["pnl"] >= 0 else "exit_loss"
            self._log(f"  EXIT ${t['exitPrice']:.2f} | {sign}${t['pnl']:.2f} | {t['exitReason']}", tag_x)

        self._log(
            f"═══ DONE: {stats['totalTrades']} trades | "
            f"WR:{stats['winRate']:.1f}% | "
            f"PnL:${stats['totalPnl']:+.2f} | "
            f"Sharpe:{stats['sharpeRatio']:.2f} ═══", "info")

    # ── SIMULATION ────────────────────────────
    def _start_sim(self):
        if self._sim_running:
            return
        self._sim_running = True
        self._btn_simulate.config(state="disabled")
        self._btn_backtest.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._clear_log()
        self._tree.delete(*self._tree.get_children())
        self._trade_summary.config(text="No trades yet", fg=MUTED)
        for k, (v, l) in self._banner_vars.items():
            v.set("—"); l.config(fg=MUTED)
        self._banner_vars["BALANCE"][0].set("$50,000.00")
        self._banner_vars["BALANCE"][1].config(fg=CYAN)
        for k, (v, l) in self._stat_frames.items():
            v.set("—")
        self._log("═══ LIVE SIMULATION v5 ═══", "info")
        self._log("Regime + Session + Pullback active", "regime")
        self._candles = generate_synthetic_data(100)
        self._bar_idx = 0
        self._engine  = VWAPStrategyEngine()
        self._equity  = []
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    def _sim_loop(self):
        import time
        settings = self._get_settings()
        while self._sim_running:
            last  = self._candles[-1]
            ret   = (random.random() - 0.5) * 0.012
            np_   = last["close"] * (1 + ret)
            rng   = np_ * (0.002 + random.random() * 0.006)
            self._candles.append({
                "time": self._bar_idx, "open": last["close"],
                "high":  round(max(last["close"], np_) + rng * random.random(), 2),
                "low":   round(min(last["close"], np_) - rng * random.random(), 2),
                "close": round(np_, 2),
                "volume": 5000 + random.random() * 45000
            })
            if len(self._candles) > 600:
                self._candles = self._candles[-600:]
            bars = calculate_all_indicators(self._candles, STRATEGY_CONFIG)
            self._bar_idx += 1
            bar = bars[-1]
            pb  = {**bars[-2], "_prevRSI": bars[-3].get("rsi", 50)} if len(bars) > 2 else bar
            pb2 = bars[-3] if len(bars) > 3 else pb
            engine = self._engine

            # Update regime display
            regime = bar.get("regime", "—")
            regime_color = {REGIME_TRENDING: GREEN, REGIME_RANGING: YELLOW,
                            REGIME_COMPRESSED: RED, REGIME_SPIKE: ORANGE}.get(regime, MUTED)
            self.after(0, lambda r=regime, c=regime_color:
                       (self._regime_var.set(f"REGIME: {r}"),
                        self._banner.winfo_children()))  # just update label

            if engine.state == STATE_IN_TRADE:
                es = engine.check_exit(bar, self._bar_idx)
                if es:
                    mode = "full"
                    if es["action"] == "partial1":
                        mode = "partial1"
                    elif es["action"] == "partial2":
                        mode = "partial2"
                    res = engine.exec_exit(bar["close"], es["reason"], self._bar_idx, mode)
                    if mode == "full":
                        tag = "exit_win" if res["pnl"] >= 0 else "exit_loss"
                        pnl = res["pnl"]
                        p   = bar["close"]
                        r   = es["reason"]
                        self.after(0, lambda pnl=pnl, p=p, r=r, tg=tag:
                                   self._log(f"  EXIT ${p:.2f} | {'+'if pnl>=0 else ''}${pnl:.2f} | {r}", tg))
                    else:
                        self.after(0, lambda r=es["reason"], p=bar["close"]:
                                   self._log(f"  {r} @ ${p:.2f}", "partial"))

            if engine.state != STATE_IN_TRADE:
                sig = engine.check_entry(bar, pb, pb2, settings, self._bar_idx)
                if sig["action"] in ("buy", "sell_short"):
                    engine.exec_entry(sig["action"], bar["close"],
                                      bar.get("atr") or bar["close"] * 0.01,
                                      sig["setup"], self._bar_idx, bar)
                    arrow = "▲" if sig["action"] == "buy" else "▼"
                    tag_e = "entry_long" if sig["action"] == "buy" else "entry_short"
                    p = bar["close"]; r = sig["reason"]
                    self.after(0, lambda a=arrow, p=p, r=r, t=tag_e:
                               self._log(f"{a} ENTRY ${p:.2f} | {r}", t))

            stats = engine.get_stats()
            self._equity.append(engine.balance)
            self.after(0, lambda s=stats, b=bars, e=engine:
                       self._sim_update(s, b, e))
            time.sleep(1.0)

    def _sim_update(self, stats, bars, engine):
        self._update_banner(stats)
        self._update_stats_row(stats, len(bars))
        self._populate_trade_table(engine.trades)
        self._update_setup_panel(stats.get("setupBreakdown", {}))
        self._draw_equity()
        self._update_position(engine)

    def _stop_sim(self):
        self._sim_running = False
        self._btn_simulate.config(state="normal")
        self._btn_backtest.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._log("═══ SIMULATION STOPPED ═══", "info")

    def _reset(self):
        self._stop_sim()
        self._bars = []; self._trades = []; self._stats = None; self._equity = []
        self._tree.delete(*self._tree.get_children())
        self._trade_summary.config(text="No trades yet", fg=MUTED)
        for k, (v, l) in self._banner_vars.items():
            v.set("—"); l.config(fg=MUTED)
        self._banner_vars["BALANCE"][0].set("$50,000.00")
        self._banner_vars["BALANCE"][1].config(fg=CYAN)
        for k, (v, l) in self._stat_frames.items():
            v.set("—")
        for k, v in self._pos_labels.items():
            v.set("—")
        self._pos_labels["Status"].set("Seeking Entry")
        self._regime_var.set("REGIME: —")
        for setup in [SETUP_VWAP_RECLAIM, SETUP_BAND_BOUNCE, SETUP_DISCOUNT, SETUP_EMA_PULLBACK]:
            for key in ["Trades", "Win%", "Net PnL"]:
                self._setup_labels[(setup, key)].set("—")
        self._clear_log()
        self._eq_canvas.delete("all")
        self._log("System reset.", "info")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = VWAPDashboard()
    app.mainloop()