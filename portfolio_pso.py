"""
Constrained Portfolio Optimization using Mean-Variance + PSO
-----------------------------------------------------------
This script is designed for a university project and demonstrates how
Computational Intelligence (PSO) can solve a realistic constrained portfolio
optimization problem that becomes difficult for classical methods.

Core objective:
- Maximize Sharpe Ratio under constraints:
  1) Budget: sum(weights) = 1
  2) Boundary: 0 <= weight_i <= 0.25
  3) Cardinality: exactly 10 stocks selected (weight > 0)

What this script produces:
- PSO-optimized portfolio vs Equal-Weight (1/30) benchmark
- 1-year backtest of $10,000
- Efficient frontier-like random portfolio cloud
- Bar chart for the 10 selected stocks
- Cumulative return comparison chart
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# =============================
# 1) Configuration
# =============================

# 30 large-cap US tickers (S&P 500 names). You can replace with SET tickers if needed.
TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "PG", "UNH", "MA", "HD", "XOM", "CVX", "LLY", "MRK",
    "ABBV", "PEP", "KO", "COST", "BAC", "AVGO", "ADBE", "CRM", "PFE", "CSCO",
]

YEARS_OF_DATA = 5
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252

# Constraint parameters
MAX_WEIGHT_PER_STOCK = 0.25
CARDINALITY_K = 10

# Strict lower bound for selected stocks (buy-in threshold).
# This removes the tiny-weight edge case and makes cardinality practically meaningful.
MIN_ACTIVE_WEIGHT = 0.05

# Active threshold used in auditing / cardinality checks.
ACTIVE_THRESHOLD = 1e-6

# PSO hyperparameters
SWARM_SIZE = 100
MAX_ITER = 500
PSO_RESTARTS = 3

# Reproducibility
RANDOM_SEED = 42


@dataclass
class PortfolioStats:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float


# =============================
# 2) Data acquisition & preprocessing
# =============================

def download_adjusted_close(tickers: List[str], years: int = 5) -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.

    Why:
    - Adjusted close includes corporate actions (splits/dividends), which makes
      returns more realistic and comparable over long periods.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Use a small buffer period for weekends/holidays
    period = f"{years}y"
    prices = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if prices.empty:
        raise RuntimeError("Failed to download data from yfinance. Check internet/ticker symbols.")

    # yfinance may return multi-index columns. We only need Close if not auto_adjusted.
    # With auto_adjust=True, 'Close' is already adjusted effectively.
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"].copy()
    else:
        close = prices.copy()

    close = close.dropna(how="all")
    close = close.ffill().dropna(how="any")

    if close.shape[1] != len(tickers):
        # Keep only columns we successfully downloaded and preserve order if possible.
        available = [t for t in tickers if t in close.columns]
        close = close[available]
        if len(available) < CARDINALITY_K:
            raise RuntimeError(
                f"Insufficient valid tickers after cleaning: {len(available)} < cardinality {CARDINALITY_K}"
            )

    print(f"Downloaded data until {end_date}: {close.shape[0]} rows x {close.shape[1]} assets")
    return close


def compute_return_inputs(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convert prices to daily returns and annualized moments.

    Why:
    - Mean-Variance model needs expected return vector (mu) and covariance matrix (Sigma).
    - Annualization aligns with annual risk-free rate for Sharpe ratio.
    """
    daily_returns = price_df.pct_change().dropna()
    mu_annual = daily_returns.mean() * TRADING_DAYS
    cov_annual = daily_returns.cov() * TRADING_DAYS
    return daily_returns, mu_annual, cov_annual


# =============================
# 3) Portfolio math helpers
# =============================

def portfolio_performance(weights: np.ndarray, mu_annual: np.ndarray, cov_annual: np.ndarray) -> PortfolioStats:
    """Compute annual return, annual volatility, Sharpe ratio."""
    annual_return = float(np.dot(weights, mu_annual))
    variance = float(weights.T @ cov_annual @ weights)
    annual_volatility = float(np.sqrt(max(variance, 1e-12)))
    sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility
    return PortfolioStats(annual_return, annual_volatility, sharpe)


def project_weights_with_cardinality(raw_weight_genes: np.ndarray, selection_genes: np.ndarray) -> np.ndarray:
    """
    Map a raw PSO particle into feasible-ish portfolio weights.

    Design choice:
    - First, choose exactly K assets using top-K selection genes.
    - Then assign positive weights on selected assets only.
    - Enforce boundary [MIN_ACTIVE_WEIGHT, MAX_WEIGHT_PER_STOCK] and normalize to sum=1.

    Why:
    - This gives PSO a smooth-ish search process while still strongly steering to exact cardinality.
    - Penalty is added later for any residual constraint violations after projection.
    """
    n_assets = len(raw_weight_genes)

    # 1) Exactly K selected indices by ranking selection genes
    ranked_idx = np.argsort(selection_genes)[::-1]
    selected_idx = ranked_idx[:CARDINALITY_K]

    # 2) Build non-negative raw weights for selected assets only
    w = np.zeros(n_assets)
    selected_raw = np.clip(raw_weight_genes[selected_idx], 1e-6, None)

    # 3) Initial normalization on selected set
    selected_sum = float(np.sum(selected_raw))
    if selected_sum <= 0:
        selected_raw = np.ones_like(selected_raw)
        selected_sum = float(np.sum(selected_raw))

    selected_w = selected_raw / selected_sum

    # 4) If minimum active weight required, apply it first, then distribute residual
    if MIN_ACTIVE_WEIGHT > 0:
        min_total = MIN_ACTIVE_WEIGHT * CARDINALITY_K
        if min_total > 1.0:
            raise ValueError("MIN_ACTIVE_WEIGHT is too high for given cardinality and budget.")
        residual = 1.0 - min_total
        selected_w = MIN_ACTIVE_WEIGHT + residual * selected_w

    # 5) Cap each selected weight by MAX_WEIGHT_PER_STOCK, then renormalize iteratively
    selected_w = np.clip(selected_w, 0.0, MAX_WEIGHT_PER_STOCK)

    # Iterative repair to satisfy sum=1 with upper bound caps
    for _ in range(20):
        total = float(np.sum(selected_w))
        if abs(total - 1.0) < 1e-9:
            break

        if total <= 0:
            selected_w = np.ones_like(selected_w) / len(selected_w)
            selected_w = np.clip(selected_w, 0.0, MAX_WEIGHT_PER_STOCK)
            continue

        # Scale uncapped components preferentially
        free_mask = selected_w < (MAX_WEIGHT_PER_STOCK - 1e-12)
        if not np.any(free_mask):
            selected_w = selected_w / total
            selected_w = np.clip(selected_w, 0.0, MAX_WEIGHT_PER_STOCK)
            continue

        delta = 1.0 - total
        free_weights = selected_w[free_mask]
        free_sum = float(np.sum(free_weights))
        if free_sum <= 1e-12:
            selected_w[free_mask] += delta / np.sum(free_mask)
        else:
            selected_w[free_mask] += delta * (free_weights / free_sum)

        selected_w = np.clip(selected_w, 0.0, MAX_WEIGHT_PER_STOCK)

    # Final normalization fallback
    total = float(np.sum(selected_w))
    if total > 0:
        selected_w /= total

    w[selected_idx] = selected_w

    # Safety cleanup for numerical noise
    w[np.abs(w) < ACTIVE_THRESHOLD] = 0.0
    return w


def constraint_penalty(weights: np.ndarray) -> float:
    """
    Penalty function for constraint handling in PSO.

    Why:
    - PSO in basic form does not natively support equality/integer-like constraints.
    - Penalty converts constraint violations into objective degradation.
    """
    penalty = 0.0

    # Budget: sum(w)=1
    budget_violation = abs(np.sum(weights) - 1.0)
    penalty += 1_000.0 * budget_violation

    # Boundary: 0 <= w_i <= MAX_WEIGHT_PER_STOCK
    lower_violation = np.sum(np.clip(0.0 - weights, 0.0, None))
    upper_violation = np.sum(np.clip(weights - MAX_WEIGHT_PER_STOCK, 0.0, None))
    penalty += 1_000.0 * (lower_violation + upper_violation)

    # Cardinality: exactly K assets with strictly positive weight
    active_count = int(np.sum(weights > ACTIVE_THRESHOLD))
    cardinality_violation = abs(active_count - CARDINALITY_K)
    penalty += 5_000.0 * cardinality_violation

    # Optional minimum active weight for selected names
    if MIN_ACTIVE_WEIGHT > 0:
        positive_weights = weights[weights > ACTIVE_THRESHOLD]
        if len(positive_weights) > 0:
            min_violation = np.sum(np.clip(MIN_ACTIVE_WEIGHT - positive_weights, 0.0, None))
            penalty += 3_000.0 * min_violation

    return penalty


# =============================
# 4) PSO optimization
# =============================

def _run_pso_single(
    objective_fn,
    lb: np.ndarray,
    ub: np.ndarray,
    swarmsize: int,
    maxiter: int,
    omega_start: float = 0.9,
    omega_end: float = 0.4,
    phip: float = 1.5,
    phig: float = 1.5,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Custom PSO loop with:
    - Linearly Decreasing Inertia Weight (omega: 0.9 -> 0.4)
      Explores broadly early, exploits precisely late.
    - Per-iteration convergence history recording.

    Returns: (best_position, best_fitness, convergence_history)
    """
    np.random.seed(seed)
    n_dim = len(lb)

    # Initialize swarm positions uniformly in [lb, ub]
    X = lb + np.random.rand(swarmsize, n_dim) * (ub - lb)
    V = np.zeros((swarmsize, n_dim))

    # Evaluate initial fitness
    fitness = np.array([objective_fn(X[i]) for i in range(swarmsize)])

    pbest_X = X.copy()
    pbest_f = fitness.copy()

    gbest_idx = int(np.argmin(pbest_f))
    gbest_X = pbest_X[gbest_idx].copy()
    gbest_f = float(pbest_f[gbest_idx])

    convergence: List[float] = []

    for it in range(maxiter):
        # Linearly decrease omega from omega_start to omega_end
        omega = omega_start - (omega_start - omega_end) * it / max(maxiter - 1, 1)

        r1 = np.random.rand(swarmsize, n_dim)
        r2 = np.random.rand(swarmsize, n_dim)

        V = (
            omega * V
            + phip * r1 * (pbest_X - X)
            + phig * r2 * (gbest_X - X)
        )
        X = np.clip(X + V, lb, ub)

        fitness = np.array([objective_fn(X[i]) for i in range(swarmsize)])

        improved = fitness < pbest_f
        pbest_X[improved] = X[improved].copy()
        pbest_f[improved] = fitness[improved]

        best_idx = int(np.argmin(pbest_f))
        if pbest_f[best_idx] < gbest_f:
            gbest_f = float(pbest_f[best_idx])
            gbest_X = pbest_X[best_idx].copy()

        # Record best Sharpe this iteration (negate because we minimized -Sharpe)
        convergence.append(-gbest_f)

    return gbest_X, gbest_f, convergence


def optimize_portfolio_pso(
    mu_annual: pd.Series, cov_annual: pd.DataFrame
) -> Tuple[np.ndarray, float, pd.DataFrame, List[List[float]]]:
    """
    Run multi-restart PSO on a mixed 2N encoding:
    - First N genes: raw positive scores for weights
    - Next N genes: selection scores to choose exactly K assets via top-K

    Returns: (best_weights, best_objective, run_history_df, convergence_histories)
    convergence_histories[i] = list of best Sharpe per iteration for run i
    """
    mu = mu_annual.values
    cov = cov_annual.values
    n_assets = len(mu)

    lb = np.zeros(2 * n_assets, dtype=float)
    ub = np.ones(2 * n_assets, dtype=float)

    def objective(x: np.ndarray) -> float:
        raw_weight_genes = x[:n_assets]
        selection_genes = x[n_assets:]
        w = project_weights_with_cardinality(raw_weight_genes, selection_genes)
        stats = portfolio_performance(w, mu, cov)
        return -stats.sharpe_ratio + constraint_penalty(w)

    print(f"Running custom PSO... (swarm={SWARM_SIZE}, iterations={MAX_ITER}, restarts={PSO_RESTARTS}, omega: 0.9->0.4)")

    best_x = None
    best_f = float("inf")
    run_rows: List[Dict[str, float]] = []
    convergence_histories: List[List[float]] = []

    for run_id in range(PSO_RESTARTS):
        seed = RANDOM_SEED + run_id
        random.seed(seed)

        xopt, fopt, conv_history = _run_pso_single(
            objective, lb, ub,
            swarmsize=SWARM_SIZE,
            maxiter=MAX_ITER,
            omega_start=0.9,
            omega_end=0.4,
            phip=1.5,
            phig=1.5,
            seed=seed,
        )

        convergence_histories.append(conv_history)

        w_run = project_weights_with_cardinality(xopt[:n_assets], xopt[n_assets:])
        stats_run = portfolio_performance(w_run, mu, cov)

        run_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "objective": float(fopt),
                "sharpe": float(stats_run.sharpe_ratio),
                "return": float(stats_run.annual_return),
                "vol": float(stats_run.annual_volatility),
            }
        )
        print(f"  Run {run_id} (seed={seed}): Sharpe={stats_run.sharpe_ratio:.4f}")

        if fopt < best_f:
            best_f = float(fopt)
            best_x = xopt

    assert best_x is not None
    best_w = project_weights_with_cardinality(best_x[:n_assets], best_x[n_assets:])
    print(f"PSO complete. Best objective value: {best_f:.6f}")
    return best_w, best_f, pd.DataFrame(run_rows), convergence_histories


def build_constraint_audit(weights: np.ndarray) -> Dict[str, float]:
    active_mask = weights > ACTIVE_THRESHOLD
    positive_weights = weights[active_mask]

    min_positive = float(np.min(positive_weights)) if len(positive_weights) > 0 else 0.0
    return {
        "sum_weights": float(np.sum(weights)),
        "max_weight": float(np.max(weights)),
        "active_count": int(np.sum(active_mask)),
        "min_active_weight": min_positive,
        "budget_ok": bool(abs(np.sum(weights) - 1.0) < 1e-6),
        "boundary_ok": bool(np.all(weights >= -1e-10) and np.all(weights <= MAX_WEIGHT_PER_STOCK + 1e-10)),
        "cardinality_ok": bool(int(np.sum(active_mask)) == CARDINALITY_K),
        "buyin_ok": bool((MIN_ACTIVE_WEIGHT <= 0.0) or (min_positive >= MIN_ACTIVE_WEIGHT - 1e-8)),
    }


# =============================
# 5) Benchmark, backtest, and plotting
# =============================

def build_equal_weight(n_assets: int) -> np.ndarray:
    """Equal weight benchmark across all assets."""
    return np.ones(n_assets) / n_assets


def backtest_cumulative_value(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float = 10_000.0,
    lookback_days: int = 252,
) -> pd.Series:
    """
    Backtest buy-and-hold weighted portfolio over the last lookback_days.

    Why:
    - Provides practical interpretation: how money grows over time.
    """
    test_returns = daily_returns.tail(lookback_days)
    portfolio_daily = test_returns.values @ weights
    cumulative = (1.0 + pd.Series(portfolio_daily, index=test_returns.index)).cumprod()
    return initial_capital * cumulative


def simulate_random_constrained_portfolios(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    n_samples: int = 5000,
) -> pd.DataFrame:
    """
    Generate random feasible portfolios (with same cardinality/boundary idea) to visualize
    risk-return opportunities (efficient-frontier-like cloud).
    """
    np.random.seed(RANDOM_SEED)

    mu = mu_annual.values
    cov = cov_annual.values
    n_assets = len(mu)

    risks = []
    rets = []
    sharpes = []

    for _ in range(n_samples):
        selected = np.random.choice(n_assets, size=CARDINALITY_K, replace=False)

        # Dirichlet gives positive weights summing to 1 on selected set
        local_w = np.random.dirichlet(np.ones(CARDINALITY_K))
        local_w = np.clip(local_w, 0.0, MAX_WEIGHT_PER_STOCK)

        # repair sum=1 after clipping
        total = local_w.sum()
        if total <= 0:
            continue
        local_w = local_w / total

        # skip if still violates cap badly after renorm
        if np.any(local_w > MAX_WEIGHT_PER_STOCK + 1e-9):
            continue

        w = np.zeros(n_assets)
        w[selected] = local_w

        stats = portfolio_performance(w, mu, cov)
        risks.append(stats.annual_volatility)
        rets.append(stats.annual_return)
        sharpes.append(stats.sharpe_ratio)

    return pd.DataFrame({"risk": risks, "return": rets, "sharpe": sharpes})


def plot_efficient_frontier(
    random_df: pd.DataFrame,
    pso_stats: PortfolioStats,
    eq_stats: PortfolioStats,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        random_df["risk"],
        random_df["return"],
        c=random_df["sharpe"],
        cmap="viridis",
        alpha=0.45,
        s=12,
        label="Random constrained portfolios",
    )
    plt.colorbar(sc, label="Sharpe Ratio")

    plt.scatter(
        pso_stats.annual_volatility,
        pso_stats.annual_return,
        marker="*",
        s=260,
        color="red",
        label="PSO Portfolio",
    )
    plt.scatter(
        eq_stats.annual_volatility,
        eq_stats.annual_return,
        marker="X",
        s=180,
        color="black",
        label="Equal Weight (1/30)",
    )

    plt.title("Efficient Frontier (Random Constrained Cloud) + PSO Solution")
    plt.xlabel("Annual Volatility (Risk)")
    plt.ylabel("Annual Expected Return")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_selected_weights(tickers: List[str], weights: np.ndarray, save_path: str | None = None) -> None:
    selected_mask = weights > ACTIVE_THRESHOLD
    selected_tickers = np.array(tickers)[selected_mask]
    selected_weights = weights[selected_mask]

    order = np.argsort(selected_weights)[::-1]
    selected_tickers = selected_tickers[order]
    selected_weights = selected_weights[order]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(selected_tickers, selected_weights)
    plt.title("PSO Selected 10 Stocks and Weights")
    plt.ylabel("Weight")
    plt.ylim(0, max(0.28, selected_weights.max() + 0.03))
    plt.grid(axis="y", alpha=0.25)

    for b, w in zip(bars, selected_weights):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{w:.2%}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cumulative_returns(pso_curve: pd.Series, eq_curve: pd.Series, save_path: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(pso_curve.index, pso_curve.values, label="PSO Portfolio", linewidth=2)
    plt.plot(eq_curve.index, eq_curve.values, label="Equal Weight Portfolio", linewidth=2)
    plt.title("Backtest: Growth of $10,000 (Last 1 Year)")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_convergence_curve(
    convergence_histories: List[List[float]],
    save_path: str | None = None,
) -> None:
    """
    Plot PSO convergence: best Sharpe Ratio per iteration for each restart.

    Why this matters:
    - A rising then flattening curve proves PSO is learning, not guessing randomly.
    - Multiple runs converging to similar values indicates algorithmic stability.
    """
    plt.figure(figsize=(10, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    best_run_idx = int(np.argmax([hist[-1] for hist in convergence_histories]))

    for i, hist in enumerate(convergence_histories):
        label = f"Run {i} (seed={42 + i})"
        alpha = 1.0 if i == best_run_idx else 0.5
        lw = 2.2 if i == best_run_idx else 1.2
        plt.plot(range(1, len(hist) + 1), hist,
                 color=colors[i % len(colors)],
                 linewidth=lw, alpha=alpha,
                 label=label + (" ← Best" if i == best_run_idx else ""))

    plt.title("PSO Convergence Curve (Best Sharpe Ratio per Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Sharpe Ratio")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================
# 6) Main workflow
# =============================

def main() -> None:
    # Step A: Data
    prices = download_adjusted_close(TICKERS, YEARS_OF_DATA)
    returns_daily, mu_annual, cov_annual = compute_return_inputs(prices)

    tickers = list(mu_annual.index)
    n_assets = len(tickers)

    # Step B: Split train/test for realistic evaluation
    # Train: all data except last 252 days; Test: last 252 days for backtest
    if len(returns_daily) <= TRADING_DAYS + 60:
        raise RuntimeError("Not enough return observations for train/test split.")

    train_returns = returns_daily.iloc[:-TRADING_DAYS].copy()
    test_returns = returns_daily.iloc[-TRADING_DAYS:].copy()

    mu_train = train_returns.mean() * TRADING_DAYS
    cov_train = train_returns.cov() * TRADING_DAYS

    # Step C: Optimize with PSO on training set
    pso_w, pso_objective, run_history_df, convergence_histories = optimize_portfolio_pso(mu_train, cov_train)

    # Step D: Benchmark (Equal Weight)
    eq_w = build_equal_weight(n_assets)

    # Step E: Evaluate both on training moments and test backtest
    pso_train_stats = portfolio_performance(pso_w, mu_train.values, cov_train.values)
    eq_train_stats = portfolio_performance(eq_w, mu_train.values, cov_train.values)

    # Backtest on out-of-sample 1-year daily returns
    pso_curve = backtest_cumulative_value(test_returns, pso_w, initial_capital=10_000.0, lookback_days=TRADING_DAYS)
    eq_curve = backtest_cumulative_value(test_returns, eq_w, initial_capital=10_000.0, lookback_days=TRADING_DAYS)

    pso_final = float(pso_curve.iloc[-1])
    eq_final = float(eq_curve.iloc[-1])

    # Step F: Random portfolios for frontier visualization
    random_df = simulate_random_constrained_portfolios(mu_train, cov_train, n_samples=5000)

    # Step G: Print report-like outputs
    print("\n" + "=" * 80)
    print("CONSTRAINED PORTFOLIO OPTIMIZATION REPORT")
    print("=" * 80)

    print("\nSelected assets by PSO (exactly 10 expected):")
    selected = [(t, w) for t, w in zip(tickers, pso_w) if w > ACTIVE_THRESHOLD]
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)
    for t, w in selected_sorted:
        print(f"  {t:>6s}: {w:7.4f} ({w:6.2%})")

    constraint_audit = build_constraint_audit(pso_w)

    print("\nConstraint check:")
    print(f"  Sum of weights: {constraint_audit['sum_weights']:.8f}")
    print(f"  Max weight:     {constraint_audit['max_weight']:.8f}")
    print(f"  Active stocks:  {constraint_audit['active_count']}")
    print(f"  Min active w:   {constraint_audit['min_active_weight']:.8f}")
    print(
        "  Checks OK:      "
        f"Budget={constraint_audit['budget_ok']}, "
        f"Boundary={constraint_audit['boundary_ok']}, "
        f"Cardinality={constraint_audit['cardinality_ok']}, "
        f"Buy-in={constraint_audit['buyin_ok']}"
    )

    print("\nTraining-period metrics (annualized):")
    print(
        f"  PSO   -> Return: {pso_train_stats.annual_return:.4f}, "
        f"Vol: {pso_train_stats.annual_volatility:.4f}, "
        f"Sharpe: {pso_train_stats.sharpe_ratio:.4f}"
    )
    print(
        f"  Equal -> Return: {eq_train_stats.annual_return:.4f}, "
        f"Vol: {eq_train_stats.annual_volatility:.4f}, "
        f"Sharpe: {eq_train_stats.sharpe_ratio:.4f}"
    )

    print("\n1-Year Backtest ($10,000 initial):")
    print(f"  PSO final value:   ${pso_final:,.2f}")
    print(f"  Equal final value: ${eq_final:,.2f}")

    # Step H: Export reproducible artifacts for reporting
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(selected_sorted, columns=["ticker", "weight"]).to_csv("outputs/pso_weights.csv", index=False)
    pd.DataFrame([constraint_audit]).to_csv("outputs/constraint_audit.csv", index=False)
    run_history_df.to_csv("outputs/pso_run_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "portfolio": "PSO",
                "objective": pso_objective,
                "ann_return": pso_train_stats.annual_return,
                "ann_vol": pso_train_stats.annual_volatility,
                "sharpe": pso_train_stats.sharpe_ratio,
                "final_value": pso_final,
            },
            {
                "portfolio": "EqualWeight",
                "objective": np.nan,
                "ann_return": eq_train_stats.annual_return,
                "ann_vol": eq_train_stats.annual_volatility,
                "sharpe": eq_train_stats.sharpe_ratio,
                "final_value": eq_final,
            },
        ]
    ).to_csv("outputs/metrics_summary.csv", index=False)

    # Step H-extra: Export convergence history
    max_len = max(len(h) for h in convergence_histories)
    conv_df = pd.DataFrame(
        {f"run_{i}_sharpe": h + [h[-1]] * (max_len - len(h)) for i, h in enumerate(convergence_histories)}
    )
    conv_df.index.name = "iteration"
    conv_df.to_csv("outputs/convergence_history.csv")

    # Step I: Visualization
    plot_efficient_frontier(
        random_df,
        pso_train_stats,
        eq_train_stats,
        save_path="outputs/efficient_frontier.png",
    )
    plot_selected_weights(tickers, pso_w, save_path="outputs/selected_weights.png")
    plot_cumulative_returns(pso_curve, eq_curve, save_path="outputs/cumulative_returns.png")
    plot_convergence_curve(convergence_histories, save_path="outputs/convergence_curve.png")


if __name__ == "__main__":
    main()
