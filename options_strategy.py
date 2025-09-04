import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import math
from scipy.interpolate import interp1d


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1.0)
    open_vol  = log_oc.pow(2).rolling(window).sum() / (window - 1.0)
    window_rs = rs.rolling(window).sum() / (window - 1.0)
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs) * math.sqrt(trading_periods)
    if return_last_only:
        return result.iloc[-1]
    return result.dropna()

def build_term_structure(days, ivs):
    days = np.asarray(days, dtype=float)
    ivs  = np.asarray(ivs, dtype=float)
    idx  = np.argsort(days)
    days = days[idx]
    ivs  = ivs[idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    def term_spline(dte):
        if dte < days[0]:
            return float(ivs[0])
        elif dte > days[-1]:
            return float(ivs[-1])
        return float(spline(dte))
    return term_spline

async def compute_recommendation_yf(raw_ticker):
    symbol = raw_ticker.upper()
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty or hist.shape[0] < 31:
            return {"status":"error", "message":"Insufficient historical price data (need ≥30 bars)."}
        hist = hist.rename(columns=str.capitalize)
        rv30 = yang_zhang(hist)
        expirations = ticker.options
        if not expirations:
            return {"status":"error", "message":"No options data found for this symbol."}
        today = datetime.date.today()
        atm_iv = {}
        straddle = None
        underlying_price = hist['Close'].iloc[-1]
        for i, exp_date in enumerate(expirations):
            try:
                calls = ticker.option_chain(exp_date).calls
                puts  = ticker.option_chain(exp_date).puts
            except Exception:
                continue
            if calls.empty or puts.empty:
                continue
            calls['diff'] = (calls['strike'] - underlying_price).abs()
            puts['diff']  = (puts['strike']  - underlying_price).abs()
            call_row = calls.loc[calls['diff'].idxmin()] if not calls.empty else None
            put_row  = puts.loc[puts['diff'].idxmin()]  if not puts.empty  else None
            if call_row is None or put_row is None:
                continue
            call_iv = call_row.get('impliedVolatility', np.nan)
            put_iv  = put_row.get('impliedVolatility', np.nan)
            if not np.isnan(call_iv) and not np.isnan(put_iv):
                atm_iv[exp_date] = (call_iv + put_iv) / 2.0
            if i == 0:
                call_bid = call_row.get('bid', np.nan)
                call_ask = call_row.get('ask', np.nan)
                put_bid  = put_row.get('bid', np.nan)
                put_ask  = put_row.get('ask', np.nan)
                call_mid = np.nan if np.isnan(call_bid) or np.isnan(call_ask) else (call_bid + call_ask)/2.0
                put_mid  = np.nan if np.isnan(put_bid)  or np.isnan(put_ask)  else (put_bid  + put_ask)/2.0
                if not np.isnan(call_mid) and not np.isnan(put_mid):
                    straddle = call_mid + put_mid
        if not atm_iv:
            return {"status":"error", "message":"Could not determine ATM IV for any expirations."}
        dtes = []
        ivs  = []
        for exp_date, iv in atm_iv.items():
            dte = (datetime.datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
            dtes.append(dte)
            ivs.append(iv)
        term_spline = build_term_structure(dtes, ivs)
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45 - min(dtes))
        iv30 = term_spline(30)
        iv30_rv30 = iv30 / rv30 if rv30 and rv30 != 0 else np.nan
        avg_volume = hist['Volume'].rolling(30).mean().dropna().iloc[-1]
        expected_move = None
        if straddle is not None and underlying_price:
            expected_move = f"{round((straddle / underlying_price) * 100, 2)}%"
        avg_volume_bool = avg_volume >= 1_500_000
        iv30_rv30_bool  = iv30_rv30 >= 1.25 if not np.isnan(iv30_rv30) else False
        ts_slope_bool   = ts_slope_0_45 <= -0.00406
        if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
            rating = "Recommended"
        elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
            rating = "Consider"
        else:
            rating = "Avoid"
        return {
            "status": "ok",
            "rating": rating,
            "avg_volume": avg_volume_bool,
            "iv30_rv30": iv30_rv30_bool,
            "ts_slope_0_45": ts_slope_bool,
            "expected_move": expected_move,
            "symbol": symbol,
            "underlying_price": underlying_price,
            "iv30": iv30,
            "rv30": rv30,
            "ts_slope_val": ts_slope_0_45,
            "avg_volume_val": avg_volume,
            "iv30_rv30_val": iv30_rv30,
        }
    except Exception as e:
        return {"status":"error", "message":str(e)}




def format_recommendation_msg(result):
    if result["status"] != "ok":
        return f"⚠️ {result.get('message','Error processing request.')}"
    check = lambda b: "✅ PASS" if b else "❌ FAIL"
    lines = [
        f"<b>{result['rating']}</b> for <i>{result['symbol']}</i>",
        f"avg_volume: {check(result['avg_volume'])} (30d avg ≥ 1.5M)",
        f"iv30/rv30: {check(result['iv30_rv30'])} (≥ 1.25)",
        f"IV term slope 0→45d: {check(result['ts_slope_0_45'])} (≤ -0.00406)",
    ]
    if result['expected_move'] is not None:
        lines.append(f"Atm Straddle Est. Move: {result['expected_move']}")
    return "\n".join(lines)