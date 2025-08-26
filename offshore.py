import os
import time
import numpy as np
import pandas as pd
import random
import cvxpy as cp
from cvxpy.error import SolverError
import scipy.optimize as sco
import scipy.stats as stats
from scipy.stats import linregress, spearmanr, kendalltau
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # for legend patches

# REPRODUCIBILITY SETTINGS 
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#  OFFSHORE USD SETTINGS 
file_path        = "/kaggle/input/timeseries12/time_series1.xlsx"
sheet_name       = "Offshore_USD"
min_weight       = 0.05
max_weight       = 0.25
min_funds        = 4
max_funds        = 20
TARGET_SUBSETS   = 30000
periods_per_year = 12

# Risk-free
rf_annual  = 0.0435
rf_monthly = (1 + rf_annual)**(1/periods_per_year) - 1

# Offshore strategy allocation bounds (no CTA)
strategy_bounds = {
    'Relative Value':    (0.15, 0.24),
    'Multi-Strategy':    (0.25, 0.30),
    'Long Short Equity': (0.15, 0.25),
    'Event Driven':      (0.10, 0.15),
    'Global Macro':      (0.15, 0.25),
}

print("🔒 Random seed set to", RANDOM_SEED)
print("🌊 OFFSHORE USD PORTFOLIO OPTIMIZATION")
print("Strategy allocation constraints:")
for strat,(lb,ub) in strategy_bounds.items():
    print(f"  {strat:17}: {lb*100:.0f}%-{ub*100:.0f}%")

# METRICS & HELPERS (corrected Sortino + positional slicing) 

def annualized_return(rets, periods_per_year):
    r = pd.Series(rets).dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).prod()
    return cum**(periods_per_year / len(r)) - 1

def annualized_volatility(rets, periods_per_year):
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(rets, rf_annual, periods_per_year, method='annual'):
    """
    method='annual'  -> Sharpe = (annualized_return - rf_annual) / annualized_volatility
    method='period'  -> classic: mean(excess_period) / std(excess_period) * sqrt(periods_per_year)
    """
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return np.nan

    if method == 'annual':
        ann_ret = annualized_return(r, periods_per_year)
        ann_vol = annualized_volatility(r, periods_per_year)
        if not np.isfinite(ann_ret) or not np.isfinite(ann_vol) or ann_vol <= 0:
            return np.nan
        return (ann_ret - rf_annual) / ann_vol

    # fallback: per-period version (what you had before)
    rf_p = (1 + rf_annual)**(1/periods_per_year) - 1
    excess = r - rf_p
    denom = excess.std(ddof=1)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return (excess.mean() / denom) * np.sqrt(periods_per_year)

def sortino_ratio(rets, rf_annual, periods_per_year):
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return np.nan
    rf_p = (1 + rf_annual)**(1/periods_per_year) - 1
    excess = r - rf_p
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    return (excess.mean() / dd) * np.sqrt(periods_per_year)

def max_drawdown_info(rets):
    r = pd.Series(rets).dropna()
    if r.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough_date = dd.idxmin()
    max_dd = dd.loc[trough_date]
    peak_level = peak.loc[trough_date]
    peak_before = wealth.loc[:trough_date]
    peak_hits = peak_before[peak_before == peak_level]
    peak_date = peak_hits.index[-1] if not peak_hits.empty else wealth.index[0]
    post = wealth.loc[trough_date:]
    recov = post[post >= peak_level]
    recovery_date = recov.index[0] if not recov.empty else np.nan
    try:
        time_to_trough = wealth.index.get_loc(trough_date) - wealth.index.get_loc(peak_date)
    except Exception:
        time_to_trough = np.nan
    if pd.isna(recovery_date):
        recovery_periods = np.nan
    else:
        try:
            recovery_periods = wealth.index.get_loc(recovery_date) - wealth.index.get_loc(trough_date)
        except Exception:
            recovery_periods = np.nan
    return max_dd, peak_date, trough_date, recovery_date, time_to_trough, recovery_periods

def alpha_beta(rets, bench, rf_annual, periods_per_year):
    y, x = pd.Series(rets), pd.Series(bench)
    y, x = y.align(x, join='inner')
    y, x = y.dropna(), x.dropna()
    if len(y) < 3:
        return np.nan, np.nan
    rf_p = (1 + rf_annual)**(1/periods_per_year) - 1
    y_ex = y - rf_p
    x_ex = x - rf_p
    mask = (~y_ex.isna()) & (~x_ex.isna())
    if mask.sum() < 3:
        return np.nan, np.nan
    xv = np.var(x_ex[mask], ddof=1)
    if not np.isfinite(xv) or xv < 1e-12:
        return y_ex[mask].mean() * periods_per_year, 0.0
    try:
        slope, intercept, *_ = stats.linregress(x_ex[mask], y_ex[mask])  # slope=beta, intercept=alpha per period
        alpha_ann = intercept * periods_per_year
        beta = slope
        return alpha_ann, beta
    except Exception:
        return np.nan, np.nan

def rolling_correlation(rets, bench, window):
    y, x = pd.Series(rets), pd.Series(bench)
    y, x = y.align(x, join='inner')
    if len(y) == 0 or len(x) == 0:
        return pd.Series(index=pd.Series(rets).index, dtype=float)
    corr = y.rolling(window).corr(x)
    return corr.reindex(pd.Series(rets).index)

def portfolio_series_with_reweight(Rdf_subset, w):
    """NaN-aware reweighting each month across available funds."""
    W = np.tile(w, (len(Rdf_subset), 1))
    mask = ~Rdf_subset.isna().values
    W_eff = W * mask
    row_sums = W_eff.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        W_eff = np.where(row_sums > 0, W_eff / row_sums, W_eff)
    port_vals = np.nansum(Rdf_subset.values * W_eff, axis=1)
    port_vals[row_sums.flatten() == 0] = np.nan
    return pd.Series(port_vals, index=Rdf_subset.index)

# Always slice by position (avoid label/position mixups)
def R_pos(sub):
    return Rdf.iloc[:, sub]

# STRESS PERIODS (auto-calibrated benchmark stress + relative stress) 

def find_stress_periods(
    bench_ret,
    method='drawdown',
    dd_threshold=None,      # if provided, use absolute threshold
    dd_q=0.10,              # AUTO: shade worst dd_q of drawdown distribution
    roll_window=6,          # months
    roll_return_threshold=None,  # if provided, absolute threshold
    roll_q=0.10,            # AUTO: shade worst roll_q of rolling returns
    vol_window=6,           # months
    vol_z=1.0,              # z-score for high vol
    expand=1,               # pad ±N months
    min_len=1,              # allow short windows in low-vol regimes
    auto=True               # use quantiles when thresholds not set
):
    x = pd.Series(bench_ret).dropna()
    idx = x.index

    def _dd_from_returns(r):
        w = (1 + r).cumprod()
        peak = w.cummax()
        dd = w / peak - 1.0
        return dd, w

    dd, w = _dd_from_returns(x)

    if method == 'drawdown':
        thr = dd_threshold
        if auto or thr is None:
            thr = dd.quantile(dd_q)
        mask = dd <= thr

    elif method == 'rolling_return':
        roll = (1 + x).rolling(roll_window).apply(lambda v: np.prod(1 + v) - 1.0, raw=False)
        thr = roll_return_threshold
        if auto or thr is None:
            thr = roll.quantile(roll_q)
        mask = roll <= thr

    elif method == 'vol':
        vol = x.rolling(vol_window).std(ddof=1)
        base = vol.rolling(max(24, vol_window * 3)).mean()
        base_std = vol.rolling(max(24, vol_window * 3)).std(ddof=1)
        z = (vol - base) / base_std
        mask = z >= vol_z

    elif method == 'composite':
        roll = (1 + x).rolling(roll_window).apply(lambda v: np.prod(1 + v) - 1.0, raw=False)
        vol = x.rolling(vol_window).std(ddof=1)
        base = vol.rolling(max(24, vol_window * 3)).mean()
        base_std = vol.rolling(max(24, vol_window * 3)).std(ddof=1)
        z = (vol - base) / base_std
        thr_dd = dd_threshold if (not auto and dd_threshold is not None) else dd.quantile(dd_q)
        thr_rr = roll_return_threshold if (not auto and roll_return_threshold is not None) else roll.quantile(roll_q)
        mask = (dd <= thr_dd) | (roll <= thr_rr) | (z >= vol_z)

    else:
        raise ValueError("method must be 'drawdown', 'rolling_return', 'vol', or 'composite'")

    mask = mask.reindex(idx, fill_value=False)

    for _ in range(expand):
        mask = mask | mask.shift(1, fill_value=False) | mask.shift(-1, fill_value=False)

    periods, in_run, start, prev_dt = [], False, None, None
    for dt, flag in mask.items():
        if flag and not in_run:
            in_run, start = True, dt
        if not flag and in_run:
            end = prev_dt
            if start is not None and end is not None:
                n = x.loc[start:end].shape[0]
                if n >= min_len:
                    periods.append((start, end))
            in_run = False
        prev_dt = dt
    if in_run and start is not None:
        periods.append((start, prev_dt))

    return mask.reindex(idx, fill_value=False), periods, dd, w

def find_relative_stress_periods(
    port_ret, bench_ret,
    window=6,        # months
    gap=None,        # absolute threshold on relative windowed return (if provided)
    gap_z=1.0,       # AUTO: shade when relative window return < -gap_z * std(rel_window)
    expand=1,
    min_len=1,
    auto=True
):
    y = pd.Series(port_ret).dropna()
    x = pd.Series(bench_ret).dropna()
    y, x = y.align(x, join='inner')
    if len(y) < window + 1:
        return pd.Series(False, index=x.index), [], None

    roll_y = (1 + y).rolling(window).apply(lambda v: np.prod(1 + v) - 1.0, raw=False)
    roll_x = (1 + x).rolling(window).apply(lambda v: np.prod(1 + v) - 1.0, raw=False)
    rel = (roll_y - roll_x).dropna()

    thr = gap
    if auto or thr is None:
        s = rel.std(ddof=1)
        if not np.isfinite(s) or s <= 1e-12:
            s = rel.abs().median() if np.isfinite(rel.abs().median()) else 0.0
        thr = -gap_z * s

    mask = rel <= thr
    mask = mask.reindex(x.index, fill_value=False)

    for _ in range(expand):
        mask = mask | mask.shift(1, fill_value=False) | mask.shift(-1, fill_value=False)

    periods, in_run, start, prev_dt = [], False, None, None
    for dt, flag in mask.items():
        if flag and not in_run:
            in_run, start = True, dt
        if not flag and in_run:
            end = prev_dt
            if start is not None and end is not None:
                n = x.loc[start:end].shape[0]
                if n >= min_len:
                    periods.append((start, end))
            in_run = False
        prev_dt = dt
    if in_run and start is not None:
        periods.append((start, prev_dt))

    return mask, periods, rel

def summarize_stress_periods(stress_periods, port_ret, bench_ret,
                             label_port='Portfolio', label_bench='Benchmark'):
    rows = []
    for (a, b) in stress_periods:
        y = pd.Series(port_ret).loc[a:b].dropna()
        x = pd.Series(bench_ret).loc[a:b].dropna()
        y, x = y.align(x, join='inner')
        if len(y) < 3:
            continue
        mdd_b, *_ = max_drawdown_info(x)
        mdd_p, *_ = max_drawdown_info(y)
        rows.append({
            'Start': a.date(), 'End': b.date(), 'Months': len(y),
            f'{label_bench} Return': (1 + x).prod() - 1,
            f'{label_port} Return': (1 + y).prod() - 1,
            f'{label_bench} MDD': mdd_b,
            f'{label_port} MDD': mdd_p,
            'Corr (in-window)': y.corr(x)
        })
    return pd.DataFrame(rows)

# LOAD PORTFOLIO RETURNS & PARSE DATES (keep NaNs) 

raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
date_row = raw.iloc[0, 2:]
dates = pd.to_datetime(date_row.values)

orig_fund_names = raw.iloc[1:, 0]
orig_strategies = raw.iloc[1:, 1]
Rdf = raw.iloc[1:, 2:].T.copy()
Rdf.index = dates

Rdf = Rdf.dropna(how='all').astype(float)

# Require at least min_obs months per fund
min_obs = 24
if min_obs is not None and min_obs > 0:
    valid_cols = Rdf.columns[Rdf.count() >= min_obs]
    Rdf = Rdf[valid_cols]

# Realign fund names and strategies to filtered Rdf columns using label alignment
try:
    fund_names = orig_fund_names.loc[Rdf.columns].astype(str).tolist()
    strategies = orig_strategies.loc[Rdf.columns].astype(str).tolist()
except KeyError:
    fund_names = orig_fund_names.reset_index(drop=True).iloc[list(range(len(Rdf.columns)))].astype(str).tolist()
    strategies = orig_strategies.reset_index(drop=True).iloc[list(range(len(Rdf.columns)))].astype(str).tolist()

# Normalize strategies to match bounds keys
def _norm_strat(s):
    s0 = str(s).strip().replace('&', 'and')
    s0 = ' '.join(s0.split())
    sl = s0.lower().replace('-', ' ').replace('/', ' ')
    if 'multi' in sl and 'strategy' in sl: return 'Multi-Strategy'
    if 'relative' in sl and 'value' in sl: return 'Relative Value'
    if ('long' in sl and 'short' in sl and 'equity' in sl) or sl.startswith('lse'): return 'Long Short Equity'
    if 'event' in sl and 'driven' in sl: return 'Event Driven'
    if 'global' in sl and 'macro' in sl: return 'Global Macro'
    return s0

strategies = [_norm_strat(s) for s in strategies]

# NaN-aware moments
T, N = Rdf.shape
mu_all = Rdf.mean(skipna=True).values

# Covariance stability with light shrinkage toward diag
cov_df = Rdf.cov(min_periods=max(3, min_obs // 4)).astype(float)
cov_df = cov_df.where(np.isfinite(cov_df), 0.0)
for i in range(len(cov_df)):
    if cov_df.iat[i, i] <= 0 or not np.isfinite(cov_df.iat[i, i]):
        cov_df.iat[i, i] = 1e-4
delta = 1e-4
cov_df = (1 - delta) * cov_df + delta * pd.DataFrame(np.diag(np.diag(cov_df)),
                                                     index=cov_df.index, columns=cov_df.columns)
cov_all = cov_df.values

# Strategy mapping (filtered universe)
fund_to_strat = {i: strategies[i] for i in range(N)}
funds_by_strategy = {s: [i for i in range(N) if fund_to_strat[i] == s] for s in strategy_bounds}

print(f"✅ Loaded {N} funds over {T} periods")
start, end = Rdf.index.min(), Rdf.index.max()

# Diagnostics
print("\n🔎 Strategy availability after filtering:")
print(pd.Series(strategies).value_counts().to_string())
print("\n🔎 funds_by_strategy counts:")
for s, (lb, ub) in strategy_bounds.items():
    cnt = len(funds_by_strategy.get(s, []))
    print(f"  {s:17}: {cnt} funds (bounds {lb:.2f}-{ub:.2f})")

#  Global FEASIBILITY checks 

sum_lb = sum(lb for lb, ub in strategy_bounds.values())
sum_ub = sum(ub for lb, ub in strategy_bounds.values())
if sum_lb > 1 + 1e-9 or sum_ub < 1 - 1e-9:
    raise RuntimeError(f"Infeasible strategy bounds: sum lower={sum_lb:.2f}, sum upper={sum_ub:.2f}")

def calculate_fund_requirements():
    req = {}
    for strat,(lb,ub) in strategy_bounds.items():
        avail_cnt = len(funds_by_strategy.get(strat, []))
        if avail_cnt == 0:
            req[strat] = (0, 0)
            continue
        mn = max(0, int(np.ceil(lb / max_weight)))
        mx = min(avail_cnt, int(np.floor(ub / min_weight)))
        req[strat] = (mn, mx)
    return req

_req = calculate_fund_requirements()
bad = [s for s,(mn,mx) in _req.items() if mn > mx]
if bad:
    raise RuntimeError(f"Infeasible fund-count windows for: {bad}. "
                       f"Try relaxing per-fund weights or strategy bounds.")
 
# SUBSET GENERATION 

def is_feasible_subset(sub):
    total_min = len(sub) * min_weight
    total_max = len(sub) * max_weight
    if total_min > 1 + 1e-9 or total_max < 1 - 1e-9:
        return False
    for strat, (lb, ub) in strategy_bounds.items():
        idxs = [i for i, f in enumerate(sub) if fund_to_strat[f] == strat]
        if not idxs:
            if lb > 1e-12:
                return False
            else:
                continue
        m = len(idxs)
        strat_min = m * min_weight
        strat_max = m * max_weight
        if strat_max < lb - 1e-9 or strat_min > ub + 1e-9:
            return False
    return True

def generate_offshore_subsets(target_subsets=5000, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    req = calculate_fund_requirements()
    combos = []
    rv_range = range(req['Relative Value'][0],    req['Relative Value'][1]    + 1)
    ms_range = range(req['Multi-Strategy'][0],    req['Multi-Strategy'][1]    + 1)
    lse_range= range(req['Long Short Equity'][0], req['Long Short Equity'][1] + 1)
    ed_range = range(req['Event Driven'][0],      req['Event Driven'][1]      + 1)
    gm_range = range(req['Global Macro'][0],      req['Global Macro'][1]      + 1)
    for rv in rv_range:
        for ms in ms_range:
            for lse in lse_range:
                for ed in ed_range:
                    for gm in gm_range:
                        total = rv + ms + lse + ed + gm
                        if min_funds <= total <= max_funds:
                            combos.append((rv, ms, lse, ed, gm))
    if not combos:
        return []
    samples_per_combo = max(1, target_subsets // max(1, len(combos)))
    subsets = []
    strat_keys = list(strategy_bounds.keys())
    seen = set()
    for combo_idx, combo in enumerate(combos):
        for sample_idx in range(samples_per_combo):
            random.seed((seed or 0) * 1000003 + combo_idx * 97 + sample_idx)
            sub = []
            infeasible = False
            for strat_key, count in zip(strat_keys, combo):
                avail = funds_by_strategy[strat_key]
                if len(avail) < count:
                    infeasible = True
                    break
                sub += (avail.copy() if count == len(avail)
                        else random.sample(avail, count))
            if infeasible:
                continue
            key = tuple(sorted(sub))
            if key in seen:
                continue
            seen.add(key)
            if len(sub) >= min_funds and is_feasible_subset(sub):
                subsets.append(sub)
                if len(subsets) >= target_subsets:
                    return subsets
    return subsets

# VALIDATION & OPTIMIZERS 

def validate_solution(sub, w):
    if w is None or abs(np.sum(w)-1) > 1e-5:
        return False
    if np.any(w < min_weight-1e-5) or np.any(w > max_weight+1e-5):
        return False
    for strat,(lb,ub) in strategy_bounds.items():
        idxs=[i for i,f in enumerate(sub) if fund_to_strat[f] == strat]
        if idxs:
            alloc = w[idxs].sum()
            if alloc < lb-1e-5 or alloc > ub+1e-5:
                return False
    return True

def project_sharpe_fast(sub):
    if not is_feasible_subset(sub):
        return None
    k = len(sub)
    mu = mu_all[sub]
    cov = cov_all[np.ix_(sub, sub)]
    vec = mu - rf_monthly
    cov_reg = cov + 1e-5 * np.eye(k)
    # warm start toward tangency
    try:
        w_star = np.linalg.solve(cov_reg, vec)
    except Exception:
        w_star, *_ = np.linalg.lstsq(cov_reg, vec, rcond=None)
    s = w_star.sum()
    w_star = (w_star / s) if s != 0 else np.ones(k) / k

    w = cp.Variable(k)
    cons = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    for strat, (lb, ub) in strategy_bounds.items():
        idxs = [i for i, f in enumerate(sub) if fund_to_strat[f] == strat]
        if idxs:
            cons += [cp.sum(w[idxs]) >= lb, cp.sum(w[idxs]) <= ub]
    objective = cp.Minimize(0.5 * cp.sum_squares(w - w_star))
    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000, verbose=False)
        ok = prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None and validate_solution(sub, w.value)
        if not ok:
            try:
                prob.solve(solver=cp.ECOS, verbose=False, max_iters=10000)
            except SolverError:
                pass
            ok = prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None and validate_solution(sub, w.value)
        if not ok:
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
            except SolverError:
                pass
            ok = prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None and validate_solution(sub, w.value)
        if ok:
            return w.value
    except SolverError:
        pass
    return None

# Sortino objective consistent with corrected downside semideviation
def solve_sortino_fast(sub):
    if not is_feasible_subset(sub):
        return None
    k = len(sub)
    Rsub = R_pos(sub)

    def obj(w):
        series = portfolio_series_with_reweight(Rsub, w).values
        series = series[~np.isnan(series)]
        if series.size < 3:
            return 1e6
        excess = series - rf_monthly
        downside = np.minimum(excess, 0.0)
        dd = np.sqrt(np.mean(downside**2)) + 1e-12
        return - (np.mean(excess) / dd) * np.sqrt(periods_per_year)

    cons = [{'type':'eq','fun':lambda w: w.sum()-1}]
    for strat,(lb,ub) in strategy_bounds.items():
        idxs=[i for i,f in enumerate(sub) if fund_to_strat[f]==strat]
        if idxs:
            cons += [
                {'type':'ineq','fun': lambda w,idxs=idxs,lb=lb: w[idxs].sum()-lb},
                {'type':'ineq','fun': lambda w,idxs=idxs,ub=ub: ub-w[idxs].sum()}
            ]
    bounds=[(min_weight,max_weight)]*k
    x0=np.ones(k)/k
    for strat,(lb,ub) in strategy_bounds.items():
        idxs=[i for i,f in enumerate(sub) if fund_to_strat[f]==strat]
        if idxs:
            targ=(lb+ub)/2
            for j in idxs:
                x0[j]=targ/len(idxs)
    x0 /= x0.sum()
    res = sco.minimize(obj, x0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'ftol':1e-6,'maxiter':200})
    if res.success and validate_solution(sub, res.x):
        return res.x
    return None

#  MAIN OPTIMIZATION (series-based scoring) 

print(f"\n🔍 Generating {TARGET_SUBSETS} subsets…")
subsets = generate_offshore_subsets(TARGET_SUBSETS, seed=RANDOM_SEED)
print(f"✅ Generated {len(subsets)} subsets")

results=[]; start_t=time.time(); last=start_t
print("\n🚀 Starting optimization…")
for i,sub in enumerate(subsets):
    now=time.time()
    if i%100==0 or now-last>15:
        elapsed=now-start_t
        eta=(len(subsets)-i)*(elapsed/max(1,i))/60
        sh_count = sum(1 for r in results if r['type']=='Sharpe')
        so_count = sum(1 for r in results if r['type']=='Sortino')
        print(f"{i+1}/{len(subsets)} | Valid: {len(results):3d} | Sharpe ok: {sh_count}, Sortino ok: {so_count}, per-subset success: {(sh_count+so_count)/max(1,i*2):.1%} | ETA: {eta:4.1f}m")
        last=now

    if not is_feasible_subset(sub):
        continue

    # Sharpe: projected target, scored by realized series with reweight
    w_sh = project_sharpe_fast(sub)
    if w_sh is not None:
        series  = portfolio_series_with_reweight(R_pos(sub), w_sh)
        sr      = sharpe_ratio(series, rf_annual, periods_per_year)
        vol_ann = annualized_volatility(series, periods_per_year)
        ret_ann = annualized_return(series, periods_per_year)
        if np.isfinite(sr):
            results.append({
                'type':'Sharpe',
                'subset':sub,
                'weights':w_sh,
                'ratio':sr,
                'ret_ann': ret_ann,
                'vol_ann': vol_ann,
            })

    # Sortino: SLSQP, scored by realized series with reweight
    w_so = solve_sortino_fast(sub)
    if w_so is not None:
        series  = portfolio_series_with_reweight(R_pos(sub), w_so)
        so      = sortino_ratio(series, rf_annual, periods_per_year)
        vol_ann = annualized_volatility(series, periods_per_year)
        ret_ann = annualized_return(series, periods_per_year)
        if np.isfinite(so):
            results.append({
                'type':'Sortino',
                'subset':sub,
                'weights':w_so,
                'ratio':so,
                'ret_ann': ret_ann,
                'vol_ann': vol_ann,
            })

print("\n✅ Optimization completed!")

# PICK MAX-SHARPE AND MAX-SORTINO 
if not results:
    raise RuntimeError("No results generated. Likely causes: strategy label mismatch, infeasible bounds, or NaN covariances.")
sharpe_candidates = [r for r in results if r['type']=='Sharpe' and np.isfinite(r['ratio'])]
if not sharpe_candidates:
    raise RuntimeError("No valid Sharpe portfolios found. Check strategy mapping, feasibility, and NaN handling.")
best_sh = max(sharpe_candidates, key=lambda x: x['ratio'])
sortino_candidates = [r for r in results if r['type']=='Sortino' and np.isfinite(r['ratio'])]
best_so = max(sortino_candidates, key=lambda x: x['ratio']) if sortino_candidates else None

# CUSTOM EXCEL BENCHMARK LOADER (with percent/decimal guard) 

custom_bench_file       = "/kaggle/input/benchmark/Benchmark Data.xlsx"  # <-- set this path
custom_bench_sheet      = "Sheet2"                                       # <-- set if different
custom_bench_date_col   = "Date"                                         # in A1
# Choose which column to use as benchmark, e.g.:
# "HFRI Fund of Funds Composite Index USD" or "HFRU Hedge Fund Composite USD Index"
custom_bench_return_col = "HFRI Fund of Funds Composite Index USD"

def load_custom_benchmark(path, sheet, date_col, return_col, start, end, Rdf_index):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom benchmark Excel not found at: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(df.columns)}")
    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not found. Available: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = pd.to_numeric(df[return_col], errors="coerce").astype(float)

    # Percent/decimal guard
    med = s.dropna().abs().median()
    if np.isfinite(med) and med > 0.5:
        s = s / 100.0

    # Assume monthly returns; consolidate to month-end and align to portfolio dates.
    s = s.loc[(s.index >= start - pd.Timedelta(days=31)) & (s.index <= end + pd.Timedelta(days=31))]
    s = s.to_period('M').groupby(level=0).last()
    s.index = s.index.to_timestamp('M')
    s = s.reindex(Rdf_index).astype(float)
    return s

chosen_bench = load_custom_benchmark(
    custom_bench_file, custom_bench_sheet, custom_bench_date_col,
    custom_bench_return_col, start, end, Rdf.index
)
best_bench_name = custom_bench_return_col
print(f"\n Chosen benchmark for reporting (from Excel): {best_bench_name}")


# REPORT AND PLOTS (auto stress + relative stress, with legends) 

def report_offshore(best, name, bench_series):
    sub, w  = best['subset'], best['weights']
    port    = portfolio_series_with_reweight(R_pos(sub), w)
    ann_ret = annualized_return(port, periods_per_year)
    ann_vol = annualized_volatility(port, periods_per_year)
    sr      = sharpe_ratio(port, rf_annual, periods_per_year)
    so      = sortino_ratio(port, rf_annual, periods_per_year)

    max_dd, peak_dt, trough_dt, rec_dt, t_to_trough, rec_periods = max_drawdown_info(port)
    alpha_ann, beta  = alpha_beta(port, bench_series, rf_annual, periods_per_year)
    corr_series = rolling_correlation(port, bench_series, periods_per_year)
    if corr_series.dropna().empty:
        y, x = port.align(bench_series, join='inner')
        corr = y.corr(x) if len(y) and len(x) else np.nan
    else:
        corr = corr_series.dropna().iloc[-1]

    print(f"\n OFFSHORE {name} ({len(sub)} funds) vs {best_bench_name}:")
    print(f"  Annualized Return      : {ann_ret*100:6.2f}%")
    print(f"  Annualized Volatility  : {ann_vol*100:6.2f}%")
    print(f"  Sharpe Ratio           : {sr:8.4f}")
    print(f"  Sortino Ratio          : {so:8.4f}")
    print(f"  Max Drawdown           : {max_dd*100:6.2f}%")
    print(f"  Peak Date              : {peak_dt}")
    print(f"  Trough Date            : {trough_dt}")
    print(f"  Time to Trough (periods): {t_to_trough}")
    print(f"  Recovery Date          : {rec_dt if pd.notna(rec_dt) else 'Not recovered'}")
    print(f"  Recovery Time (periods): {rec_periods}")
    print(f"  Annual Alpha           : {alpha_ann*100 if pd.notna(alpha_ann) else np.nan:.2f}%")
    print(f"  Beta vs {best_bench_name:15s}: {beta if pd.notna(beta) else np.nan:6.3f}")
    print(f"  Corr ({periods_per_year}-mo)      : {corr if pd.notna(corr) else np.nan:6.3f}")

    print("  Strategy Allocations:")
    for strat,(lb,ub) in strategy_bounds.items():
        alloc = sum(w[j] for j,f in enumerate(sub) if fund_to_strat[f]==strat)
        status = "✅" if lb-1e-8 <= alloc <= ub+1e-8 else ""
        print(f"    {strat:17}: {alloc*100:6.2f}% ∈ [{lb*100:5.2f}%, {ub*100:5.2f}%] {status}")

    print("  Top Holdings:")
    for i,(idx,weight) in enumerate(sorted(zip(sub,w), key=lambda x: x[1], reverse=True)[:10],1):
        print(f"    {i:2d}. {fund_names[idx]:40} ({fund_to_strat[idx]:17}) {weight*100:5.2f}%")

def plot_cum_and_corr(best, name, bench_series, label_bench, periods_per_year=12,
                      show_stress=True, stress_kwargs=None,
                      show_relative_stress=True, relative_kwargs=None,
                      annotate_troughs=True):
    sub, w = best['subset'], best['weights']

    # Portfolio returns (positional slice)
    port = portfolio_series_with_reweight(R_pos(sub), w)

    # Align with benchmark
    y, x = port.align(bench_series, join='inner')
    if y.dropna().empty or x.dropna().empty:
        print(f" Skipping plots for {name}: no overlapping non-NaN data with benchmark.")
        return
    y = y.dropna(); x = x.dropna()
    idx = y.index.intersection(x.index)
    y = y.loc[idx]; x = x.loc[idx]

    wealth_p = (1 + y).cumprod(); wealth_b = (1 + x).cumprod()
    wealth_p /= wealth_p.iloc[0]; wealth_b /= wealth_b.iloc[0]
    roll_corr = y.rolling(periods_per_year).corr(x)

    # Benchmark stress (auto)
    periods = []; dd_bench = None
    if show_stress:
        if stress_kwargs is None:
            stress_kwargs = dict(method='drawdown', auto=True, dd_q=0.10, min_len=1, expand=1)
        _, periods, dd_bench, _wealth_b = find_stress_periods(x, **stress_kwargs)

    # Relative stress
    rel_periods = []
    if show_relative_stress:
        if relative_kwargs is None:
            relative_kwargs = dict(window=6, gap=None, gap_z=1.0, expand=1, min_len=1, auto=True)
        _, rel_periods, _ = find_relative_stress_periods(y, x, **relative_kwargs)

    # Plot
    fig = plt.figure(figsize=(14, 5))

    # Left: cumulative growth
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(wealth_p.index, wealth_p.values, label=name, lw=2)
    ax1.plot(wealth_b.index, wealth_b.values, label=label_bench, lw=2)

    if show_stress and periods:
        for (a, b) in periods:
            ax1.axvspan(a, b, color='lightcoral', alpha=0.18, lw=0)
        if annotate_troughs and dd_bench is not None:
            for (a, b) in periods:
                seg = dd_bench.loc[a:b]
                if not seg.empty:
                    t = seg.idxmin()
                    ax1.axvline(t, lw=1, alpha=0.35)  # trough marker

    if show_relative_stress and rel_periods:
        for (a, b) in rel_periods:
            ax1.axvspan(a, b, color='lightblue', alpha=0.18, lw=0)

    # Legend for shades (left)
    legend_patches = []
    if show_stress and periods:
        legend_patches.append(Patch(facecolor='lightcoral', alpha=0.18, label='Benchmark stress'))
    if show_relative_stress and rel_periods:
        legend_patches.append(Patch(facecolor='lightblue', alpha=0.18, label='Relative stress (underperf.)'))
    h1, l1 = ax1.get_legend_handles_labels()
    if legend_patches:
        ax1.legend(handles=h1 + legend_patches, loc='best')
    else:
        ax1.legend(loc='best')

    ax1.set_title(f'{name} vs {label_bench} - Cumulative Growth')
    ax1.set_ylabel('Growth (normalized)'); ax1.set_xlabel('Date')
    ax1.grid(True, alpha=0.3)

    # Right: rolling correlation
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(roll_corr.index, roll_corr.values, lw=2, label='Rolling corr')
    ax2.axhline(0.0, lw=1, alpha=0.6, label='Zero line')

    if show_stress and periods:
        for (a, b) in periods:
            ax2.axvspan(a, b, color='lightcoral', alpha=0.18, lw=0)
    if show_relative_stress and rel_periods:
        for (a, b) in rel_periods:
            ax2.axvspan(a, b, color='lightblue', alpha=0.18, lw=0)

    # Legend for shades (right)
    legend_patches_r = []
    if show_stress and periods:
        legend_patches_r.append(Patch(facecolor='lightcoral', alpha=0.18, label='Benchmark stress'))
    if show_relative_stress and rel_periods:
        legend_patches_r.append(Patch(facecolor='lightblue', alpha=0.18, label='Relative stress (underperf.)'))
    h2, l2 = ax2.get_legend_handles_labels()
    if legend_patches_r:
        ax2.legend(handles=h2 + legend_patches_r, loc='best')
    else:
        ax2.legend(loc='best')

    ax2.set_title(f'{name} vs {label_bench} - Rolling {periods_per_year}-Month Correlation')
    ax2.set_ylabel('Correlation'); ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

# Report/plot Max-Sortino if available
if best_so is not None:
    report_offshore(best_so, "Max-Sortino", chosen_bench)
    plot_cum_and_corr(best_so, "Max-Sortino", chosen_bench, best_bench_name, periods_per_year=periods_per_year)

# Report/plot Max-Sharpe
report_offshore(best_sh, "Max-Sharpe", chosen_bench)
plot_cum_and_corr(best_sh, "Max-Sharpe", chosen_bench, best_bench_name, periods_per_year=periods_per_year)

#  SUMMARY PLOTS: SHARPE AND SORTINO PORTFOLIOS 

if results:
    df = pd.DataFrame([{
        'Type': r['type'],
        'Ratio': r['ratio'],
        'Return (%)': r['ret_ann']*100,
        'Vol (%)': r['vol_ann']*100,
        'Fund Count': len(r['subset'])
    } for r in results])

    # SHARPE PORTFOLIOS GRAPH
    sharpe_df = df[df['Type'] == 'Sharpe']
    if len(sharpe_df) > 0:
        plt.figure(figsize=(15, 10))
        plt.suptitle('🔵 OFFSHORE USD - SHARPE OPTIMIZED PORTFOLIOS', fontsize=16, fontweight='bold')
        plt.subplot(2, 3, 1)
        plt.scatter(sharpe_df['Vol (%)'], sharpe_df['Return (%)'], c='blue', s=40, alpha=0.7)
        plt.xlabel("Volatility (%)"); plt.ylabel("Return (%)")
        plt.title("Risk-Return Profile"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 2)
        plt.scatter(sharpe_df['Fund Count'], sharpe_df['Ratio'], c='blue', s=40, alpha=0.7)
        plt.xlabel("Number of Funds"); plt.ylabel("Sharpe Ratio")
        plt.title("Sharpe Ratio vs Fund Count"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 3)
        plt.hist(sharpe_df['Ratio'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Sharpe Ratio"); plt.ylabel("Frequency")
        plt.title("Sharpe Ratio Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 4)
        plt.hist(sharpe_df['Return (%)'], bins=20, color='lightblue', alpha=0.7, edgecolor='black')
        plt.xlabel("Annual Return (%)"); plt.ylabel("Frequency")
        plt.title("Return Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 5)
        plt.hist(sharpe_df['Vol (%)'], bins=20, color='darkblue', alpha=0.7, edgecolor='black')
        plt.xlabel("Annual Volatility (%)"); plt.ylabel("Frequency")
        plt.title("Volatility Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 6)
        plt.hist(sharpe_df['Fund Count'], bins=range(min_funds, max_funds+2),
                 color='navy', alpha=0.7, edgecolor='black')
        plt.xlabel("Number of Funds"); plt.ylabel("Frequency")
        plt.title("Fund Count Distribution"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # SORTINO PORTFOLIOS GRAPH
    sortino_df = df[df['Type'] == 'Sortino']
    if len(sortino_df) > 0:
        plt.figure(figsize=(15, 10))
        plt.suptitle('🟠 OFFSHORE USD - SORTINO OPTIMIZED PORTFOLIOS', fontsize=16, fontweight='bold')
        plt.subplot(2, 3, 1)
        plt.scatter(sortino_df['Vol (%)'], sortino_df['Return (%)'], c='orange', s=40, alpha=0.7)
        plt.xlabel("Volatility (%)"); plt.ylabel("Return (%)")
        plt.title("Risk-Return Profile"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 2)
        plt.scatter(sortino_df['Fund Count'], sortino_df['Ratio'], c='orange', s=40, alpha=0.7)
        plt.xlabel("Number of Funds"); plt.ylabel("Sortino Ratio")
        plt.title("Sortino Ratio vs Fund Count"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 3)
        plt.hist(sortino_df['Ratio'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        plt.xlabel("Sortino Ratio"); plt.ylabel("Frequency")
        plt.title("Sortino Ratio Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 4)
        plt.hist(sortino_df['Return (%)'], bins=20, color='moccasin', alpha=0.7, edgecolor='black')
        plt.xlabel("Annual Return (%)"); plt.ylabel("Frequency")
        plt.title("Return Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 5)
        plt.hist(sortino_df['Vol (%)'], bins=20, color='darkorange', alpha=0.7, edgecolor='black')
        plt.xlabel("Annual Volatility (%)"); plt.ylabel("Frequency")
        plt.title("Volatility Distribution"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 6)
        plt.hist(sortino_df['Fund Count'], bins=range(min_funds, max_funds+2),
                 color='chocolate', alpha=0.7, edgecolor='black')
        plt.xlabel("Number of Funds"); plt.ylabel("Frequency")
        plt.title("Fund Count Distribution"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

#  K-COMPARISON VS BENCHMARK (auto stress shading + legend) 

def plot_k_comparison_vs_benchmark(results, Rdf, bench_series, ks=(6,8,10,12,14,16,18,20),
                                   label_bench=best_bench_name,
                                   show_stress=True, stress_kwargs=None):
    candidates = {}
    for k in ks:
        sharpe_k = [r for r in results if r['type']=='Sharpe' and len(r['subset'])==k]
        sortino_k = [r for r in results if r['type']=='Sortino' and len(r['subset'])==k]
        best_k = None
        if sharpe_k:
            best_k = max(sharpe_k, key=lambda x: x['ratio'])
        elif sortino_k:
            best_k = max(sortino_k, key=lambda x: x['ratio'])
        if best_k is not None:
            candidates[k] = best_k

    if not candidates:
        print("⚠️ No portfolios found for the requested k values. Skipping k-comparison plot.")
        return

    plt.figure(figsize=(14, 8))
    plt.title('Offshore USD - Best Portfolio vs Benchmark by Fund Count (k)')

    bench = bench_series.dropna()

    # Shade auto-calibrated stress periods once (benchmark only)
    periods = []
    if show_stress:
        if stress_kwargs is None:
            stress_kwargs = dict(method='drawdown', auto=True, dd_q=0.10, min_len=1, expand=1)
        _, periods, _, _ = find_stress_periods(bench, **stress_kwargs)
        for (a, b) in periods:
            plt.axvspan(a, b, color='lightcoral', alpha=0.18, lw=0)

    for k, best_k in sorted(candidates.items()):
        sub = best_k['subset']; w = best_k['weights']
        port = portfolio_series_with_reweight(Rdf.iloc[:, sub], w).dropna()
        y, x = port.align(bench, join='inner')
        if len(y) < 12:
            print(f"⚠️ Skipping k={k}: insufficient overlap with benchmark.")
            continue
        wealth_p = (1 + y).cumprod()
        wealth_b = (1 + x).cumprod()
        wealth_p /= wealth_p.iloc[0]
        wealth_b /= wealth_b.iloc[0]
        plt.plot(wealth_p.index, wealth_p.values, lw=2, label=f'k={k} (best)')

    # Plot benchmark once
    wb = (1 + bench).cumprod()
    wb /= wb.iloc[0]
    plt.plot(wb.index, wb.values, lw=2.5, color='black', alpha=0.7, label=f'{label_bench}')

    # Legend for shades (k-comparison)
    legend_patches = []
    if show_stress and periods:
        legend_patches.append(Patch(facecolor='lightcoral', alpha=0.18, label='Benchmark stress'))
    h, l = plt.gca().get_legend_handles_labels()
    if legend_patches:
        plt.legend(handles=h + legend_patches, loc='best')
    else:
        plt.legend(loc='best')

    plt.ylabel('Cumulative Growth (normalized)'); plt.xlabel('Date')
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Draw the k-comparison plot
plot_k_comparison_vs_benchmark(results, Rdf, chosen_bench, ks=(6,8,10,12,14,16,18,20), label_bench=best_bench_name)

# Stress-period comparison tables 

SHOW_STRESS_TABLES = True
if SHOW_STRESS_TABLES:
    # Define auto stress windows from the benchmark
    _, periods_auto, _, _ = find_stress_periods(chosen_bench, method='drawdown', auto=True, dd_q=0.10, min_len=1, expand=1)

    # Max-Sharpe table
    sub_sh, w_sh = best_sh['subset'], best_sh['weights']
    port_sh = portfolio_series_with_reweight(R_pos(sub_sh), w_sh)
    sh_tbl = summarize_stress_periods(periods_auto, port_sh, chosen_bench, label_port='Max-Sharpe', label_bench=best_bench_name)
    if not sh_tbl.empty:
        print("\n📉 Stress-window comparison: Max-Sharpe vs Benchmark")
        print(sh_tbl.to_string(index=False, float_format=lambda v: f"{v:.2%}" if isinstance(v, float) else str(v)))

    # Max-Sortino table (if available)
    if best_so is not None:
        sub_so, w_so = best_so['subset'], best_so['weights']
        port_so = portfolio_series_with_reweight(R_pos(sub_so), w_so)
        so_tbl = summarize_stress_periods(periods_auto, port_so, chosen_bench, label_port='Max-Sortino', label_bench=best_bench_name)
        if not so_tbl.empty:
            print("\n📉 Stress-window comparison: Max-Sortino vs Benchmark")
            print(so_tbl.to_string(index=False, float_format=lambda v: f"{v:.2%}" if isinstance(v, float) else str(v)))

# REGRESSION / TREND TESTS: "Fewer funds = better result?" 

if results:
    # Build df if not already built
    df = pd.DataFrame([{
        'Type': r['type'],
        'Ratio': r['ratio'],
        'Return (%)': r['ret_ann']*100,
        'Vol (%)': r['vol_ann']*100,
        'Fund Count': len(r['subset'])
    } for r in results])

    def _clean_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def regression_summary(x, y, label):
        x, y = _clean_xy(x, y)
        if x.size < 3:
            print(f"\n=== {label} ===\nNot enough points (n={x.size}).")
            return None
        lr = linregress(x, y)  # slope, intercept, rvalue, pvalue, stderr
        sp = spearmanr(x, y, nan_policy='omit')
        kt = kendalltau(x, y, nan_policy='omit')
        print(f"\n=== Linear trend: {label} ===")
        print(f"n={x.size}, slope per +1 fund = {lr.slope:.6f}")
        print(f"intercept = {lr.intercept:.6f}, R^2 = {lr.rvalue**2:.4f}")
        print(f"OLS p-value (H0: slope=0) = {lr.pvalue:.4g}")
        print(f"Spearman rho = {sp.correlation:.4f} (p={sp.pvalue:.4g})")
        print(f"Kendall tau  = {kt.correlation:.4f} (p={kt.pvalue:.4g})")
        return lr

    def permutation_pvalue(x, y, slope_obs, n_perm=2000, seed=RANDOM_SEED):
        # 2-sided permutation test: shuffle y relative to x
        rng = np.random.default_rng(seed)
        x, y = _clean_xy(x, y)
        if x.size < 3:
            return np.nan
        cnt = 0
        for _ in range(n_perm):
            s = linregress(x, rng.permutation(y)).slope
            if abs(s) >= abs(slope_obs):
                cnt += 1
        return (cnt + 1) / (n_perm + 1)

    metric = 'Ratio'

    # 1) All portfolios
    lr_all = regression_summary(df['Fund Count'], df[metric], f"All portfolios ({metric} ~ Fund Count)")
    if lr_all is not None:
        p_perm = permutation_pvalue(df['Fund Count'], df[metric], lr_all.slope, n_perm=2000)
        print(f"Permutation p-value (2-sided) = {p_perm:.4g}")

    try:
        xg = np.linspace(df['Fund Count'].min(), df['Fund Count'].max(), 100)
        plt.figure(figsize=(7,5))
        plt.scatter(df['Fund Count'], df[metric], alpha=0.35)
        if lr_all is not None:
            plt.plot(xg, lr_all.intercept + lr_all.slope * xg, lw=2)
        plt.title(f"{metric} vs Fund Count (All portfolios)")
        plt.xlabel("Fund Count"); plt.ylabel(metric)
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    except Exception:
        pass

    # 2) Best-per-k frontier (Sharpe and Sortino separately)
    best_per_k = (df.groupby(['Type','Fund Count'])[metric].max()
                    .reset_index().sort_values(['Type','Fund Count']))

    for typ in best_per_k['Type'].unique():
        sub = best_per_k[best_per_k['Type'] == typ]
        lr = regression_summary(sub['Fund Count'], sub[metric], f"Best-per-k frontier ({typ}, metric={metric})")
        if lr is not None:
            p_perm = permutation_pvalue(sub['Fund Count'], sub[metric], lr.slope, n_perm=2000)
            print(f"Permutation p-value (2-sided) = {p_perm:.4g}")
        # Optional plot
        try:
            x, y = _clean_xy(sub['Fund Count'], sub[metric])
            if x.size >= 2:
                lr_tmp = linregress(x, y)
                xg = np.linspace(x.min(), x.max(), 100)
                plt.figure(figsize=(7,5))
                plt.scatter(x, y)
                plt.plot(xg, lr_tmp.intercept + lr_tmp.slope * xg, lw=2)
                plt.title(f"Best-per-k Frontier ({typ}) — {metric} vs Fund Count")
                plt.xlabel("Fund Count"); plt.ylabel(f"Best {metric} by k")
                plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        except Exception:
            pass

    # 3) Pooled frontier (max across Sharpe/Sortino for each k)
    frontier_all = (best_per_k.groupby('Fund Count')[metric].max()
                    .reset_index().sort_values('Fund Count'))
    lr_front = regression_summary(frontier_all['Fund Count'], frontier_all[metric],
                                  f"Best-per-k frontier (pooled types, metric={metric})")
    if lr_front is not None:
        p_perm = permutation_pvalue(frontier_all['Fund Count'], frontier_all[metric], lr_front.slope, n_perm=2000)
        print(f"Permutation p-value (2-sided) = {p_perm:.4g}")

    # Optional plot
    try:
        x, y = _clean_xy(frontier_all['Fund Count'], frontier_all[metric])
        if x.size >= 2:
            lr_tmp = linregress(x, y)
            xg = np.linspace(x.min(), x.max(), 100)
            plt.figure(figsize=(7,5))
            plt.scatter(x, y)
            plt.plot(xg, lr_tmp.intercept + lr_tmp.slope * xg, lw=2)
            plt.title(f"Pooled Best-per-k Frontier — {metric} vs Fund Count")
            plt.xlabel("Fund Count"); plt.ylabel(f"Pooled Best {metric} by k")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    except Exception:
        pass
