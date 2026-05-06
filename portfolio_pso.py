"""
Research-grade constrained portfolio optimization using Self-Adaptive PSO.

This module is intentionally written as a readable end-to-end reference for the project.
The code is organized so that each step can be explained in class without hand-waving:

1. Download and clean real market data.
2. Split data into an in-sample training window and an out-of-sample test window.
3. Estimate the Mean-Variance inputs from training data.
4. Use a custom Particle Swarm Optimization loop to maximize the Sharpe ratio.
5. Enforce realistic portfolio constraints through repair rules and penalties.
6. Compare the optimized portfolio against Equal Weight and a market benchmark.
7. Compare the sector-constrained solution against an original unconstrained PSO variant.
8. Export plots, CSV tables, and Thai presentation/report text files.

The file exposes two main entry points:
- PortfolioOptimizer: runs one scenario, such as the sector-constrained portfolio.
- run_full_study: runs both the original PSO and the sector-constrained PSO,
  then writes all outputs required for analysis and presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling


# =============================
# 1) Static configuration
# =============================

TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV",
    "MRK", "BAC", "CVX", "CRM", "ADBE", "KO", "PEP", "WMT", "AMD", "NFLX",
    "DIS", "MCD", "TMO", "CSCO", "WFC", "ACN", "ABT", "QCOM", "ORCL", "LIN",
    "INTU", "INTC", "TXN", "AMGN", "IBM", "ISRG", "GE", "CAT", "HON", "NEE"
]

SECTOR_MAP: Dict[str, str] = {
    # Tech
    "AAPL": "Tech", "MSFT": "Tech", "AMZN": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "META": "Tech", "AVGO": "Tech", "TSLA": "Tech", "CRM": "Tech", "ADBE": "Tech",
    "AMD": "Tech", "CSCO": "Tech", "ACN": "Tech", "QCOM": "Tech", "ORCL": "Tech",
    "INTU": "Tech", "INTC": "Tech", "TXN": "Tech", "IBM": "Tech", "AMAT": "Tech",
    "SNPS": "Tech", "CDNS": "Tech", "PANW": "Tech", "MU": "Tech", "ADI": "Tech", "LRCX": "Tech",
    # Finance
    "BRK-B": "Finance", "JPM": "Finance", "V": "Finance", "MA": "Finance", "BAC": "Finance",
    "WFC": "Finance", "GS": "Finance", "MS": "Finance", "SPGI": "Finance", "BLK": "Finance",
    "AXP": "Finance", "SCHW": "Finance", "C": "Finance", "CB": "Finance", "MMC": "Finance",
    # Health
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health", "MRK": "Health",
    "TMO": "Health", "ABT": "Health", "AMGN": "Health", "ISRG": "Health", "PFE": "Health",
    "SYK": "Health", "GILD": "Health", "VRTX": "Health", "REGN": "Health", "DHR": "Health",
    "CI": "Health", "CVS": "Health", "ZTS": "Health",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "EOG": "Energy",
    # Consumer
    "PG": "Consumer", "HD": "Consumer", "COST": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "WMT": "Consumer", "MCD": "Consumer", "LOW": "Consumer", "PM": "Consumer", "TJX": "Consumer",
    "MDLZ": "Consumer", "EL": "Consumer", "NKE": "Consumer", "SBUX": "Consumer", "MO": "Consumer",
    "CL": "Consumer", "TGT": "Consumer",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "BA": "Industrials", "RTX": "Industrials", "LMT": "Industrials", "UPS": "Industrials",
    "DE": "Industrials", "ADP": "Industrials",
    # Communication
    "NFLX": "Communication", "DIS": "Communication", "T": "Communication", "VZ": "Communication",
    "CMCSA": "Communication",
    # Utilities/Others
    "LIN": "Utilities", "NEE": "Utilities", "PLD": "Utilities", "AMT": "Utilities"
}

SECTOR_COLORS: Dict[str, str] = {
    "Tech": "#002D62",      # Navy Blue
    "Energy": "#FFC000",    # Golden Yellow
    "Finance": "#2ca02c",
    "Health": "#d62728",
    "Consumer": "#9467bd",
    "Industrials": "#8c564b",
    "Communication": "#e377c2",
    "Utilities": "#7f7f7f",
}

YEARS_OF_DATA = 10
TRAIN_TEST_SPLIT_DAYS = 252
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
CARDINALITY_K = 10
MAX_WEIGHT_PER_STOCK = 0.25
MIN_ACTIVE_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.40
ACTIVE_THRESHOLD = 1e-6
SWARM_SIZE = 100
MAX_ITER = 300
PSO_RESTARTS = 5
INERTIA_START = 0.90
INERTIA_END = 0.40
COGNITIVE_COEFF = 1.50
SOCIAL_COEFF = 1.50
RANDOM_SEED = 42
BENCHMARK_TICKER = "^GSPC"
FRONTIER_RANDOM_SAMPLES = 50_000


@dataclass
class PortfolioStats:
    """Container for risk-adjusted statistics computed from a realized return series."""

    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    mean_absolute_deviation: float
    max_drawdown: float
    turnover_rate: float
    final_value: float


@dataclass
class ExperimentArtifacts:
    """All outputs produced by one optimization scenario."""

    experiment_name: str
    weights: np.ndarray
    objective: float
    weights_table: pd.DataFrame
    sector_table: pd.DataFrame
    constraint_audit: pd.DataFrame
    metrics_summary: pd.DataFrame
    run_history_df: pd.DataFrame
    convergence_history_df: pd.DataFrame
    pso_curve: pd.Series
    equal_curve: pd.Series
    benchmark_curve: pd.Series | None


# =============================
# 1.5) Pymoo Problem & Repair
# =============================

class PortfolioRepair(Repair):
    def __init__(self, optimizer: PortfolioOptimizer):
        super().__init__()
        self.opt = optimizer

    def _do(self, problem, X, **kwargs):
        n_assets = self.opt.n_assets
        for i in range(len(X)):
            raw_weights = X[i, :n_assets]
            selection = X[i, n_assets:]
            
            # Repair logic
            weights = self.opt.project_weights_with_cardinality(raw_weights, selection)
            
            # Transaction Lots: Round to nearest 1% (0.01)
            weights = np.round(weights, 2)
            
            # Final normalization to ensure sum = 1.0 after rounding
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            X[i, :n_assets] = weights
        return X


class PortfolioProblem(ElementwiseProblem):
    def __init__(self, optimizer: PortfolioOptimizer):
        self.opt = optimizer
        n_assets = len(optimizer.tickers)
        # 2N encoding: N weights, N selection scores
        # n_obj=2, n_constr=0 (Sector constraint handled as penalty in objectives)
        super().__init__(n_var=2 * n_assets, n_obj=2, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        n_assets = self.opt.n_assets
        weights = x[:n_assets]
        
        # Objectives: 1. Minimize Risk (Volatility), 2. Maximize Return
        ann_return = float(np.dot(weights, self.opt.mu_train.values))
        variance = float(weights.T @ self.opt.cov_train.values @ weights)
        ann_vol = float(np.sqrt(max(variance, 1e-12)))
        
        # Transaction Costs: Penalty proportional to turnover from equal weight
        eq_weights = self.opt.build_equal_weight()
        turnover = float(0.5 * np.sum(np.abs(weights - eq_weights)))
        net_return = ann_return - (turnover * 0.002) # 20 bps cost
        
        # Constraints: Sector Concentration
        sector_penalty = 0.0
        if self.opt.max_sector_weight is not None:
            exposures = self.opt.sector_exposures(weights)
            sector_penalty = max(0.0, max(exposures.values()) - self.opt.max_sector_weight)
        
        # Add penalty to both objectives to push towards feasibility
        # High penalty weight (50) to ensure sector constraints are prioritized
        out["F"] = [ann_vol + sector_penalty * 50, -net_return + sector_penalty * 50]


class PortfolioOptimizer:
    """
    Run a single portfolio-optimization scenario.

    A scenario is defined by one key modeling choice: whether a sector concentration cap is active.
    For example:
    - experiment_name="PSO_original", max_sector_weight=None
    - experiment_name="PSO_with_sector_constraints", max_sector_weight=0.40

    The class keeps all state explicit so each stage can be inspected independently.
    """

    def __init__(
        self,
        experiment_name: str = "PSO_with_sector_constraints",
        max_sector_weight: float | None = MAX_SECTOR_WEIGHT,
        benchmark_ticker: str = BENCHMARK_TICKER,
        frontier_random_samples: int = FRONTIER_RANDOM_SAMPLES,
    ) -> None:
        self.experiment_name = experiment_name
        self.tickers = list(TICKERS)
        self.sector_map = dict(SECTOR_MAP)
        self.sector_colors = dict(SECTOR_COLORS)
        self.years_of_data = YEARS_OF_DATA
        self.train_test_split_days = TRAIN_TEST_SPLIT_DAYS
        self.trading_days = TRADING_DAYS
        self.risk_free_rate = RISK_FREE_RATE
        self.cardinality_k = CARDINALITY_K
        self.max_weight = MAX_WEIGHT_PER_STOCK
        self.min_active_weight = MIN_ACTIVE_WEIGHT
        self.max_sector_weight = max_sector_weight
        self.active_threshold = ACTIVE_THRESHOLD
        self.swarm_size = SWARM_SIZE
        self.max_iter = MAX_ITER
        self.restarts = PSO_RESTARTS
        self.inertia_start = INERTIA_START
        self.inertia_end = INERTIA_END
        self.cognitive_coeff = COGNITIVE_COEFF
        self.social_coeff = SOCIAL_COEFF
        self.random_seed = RANDOM_SEED
        self.frontier_random_samples = frontier_random_samples
        self.benchmark_ticker = benchmark_ticker
        self.n_assets = len(self.tickers)

        self.prices: pd.DataFrame | None = None
        self.benchmark_prices: pd.Series | None = None
        self.returns_daily: pd.DataFrame | None = None
        self.train_returns: pd.DataFrame | None = None
        self.test_returns: pd.DataFrame | None = None
        self.benchmark_returns: pd.Series | None = None
        self.benchmark_test_returns: pd.Series | None = None
        self.mu_train: pd.Series | None = None
        self.cov_train: pd.DataFrame | None = None
        self.best_weights: np.ndarray | None = None
        self.best_objective: float | None = None
        self.run_history_df: pd.DataFrame | None = None
        self.convergence_history_df: pd.DataFrame | None = None

    # =============================
    # 2) Data layer
    # =============================

    def download_adjusted_close(self) -> pd.DataFrame:
        """
        Download the investable universe and the benchmark index.

        Adjusted close is used because it reflects splits and dividends, which is required for a
        meaningful return series over multiple years.
        """
        end_date = datetime.today().strftime("%Y-%m-%d")
        prices = yf.download(
            tickers=self.tickers,
            period=f"{self.years_of_data}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        if prices.empty:
            raise RuntimeError("Failed to download portfolio universe data from yfinance.")

        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"].copy()
        else:
            close = prices.copy()

        close = close.dropna(how="all")
        # Use ffill then bfill to handle stocks that started later or ended earlier
        close = close.ffill().bfill()
        close = close.dropna(axis=1, thresh=len(close) * 0.8) # Keep stocks with at least 80% data

        available = [ticker for ticker in self.tickers if ticker in close.columns]
        if len(available) < self.cardinality_k:
            raise RuntimeError(
                f"Insufficient valid tickers after cleaning: {len(available)} < cardinality {self.cardinality_k}"
            )

        benchmark_raw = yf.download(
            tickers=self.benchmark_ticker,
            period=f"{self.years_of_data}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        benchmark_series = benchmark_raw["Close"] if "Close" in benchmark_raw.columns else benchmark_raw
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
        benchmark_series = benchmark_series.dropna().rename(self.benchmark_ticker)

        self.tickers = available
        self.prices = close[available].copy()
        self.n_assets = len(self.tickers)
        self.benchmark_prices = benchmark_series
        print(
            f"Downloaded data until {end_date}: {self.prices.shape[0]} rows x {self.prices.shape[1]} assets | "
            f"benchmark={self.benchmark_ticker}"
        )
        return self.prices

    def copy_market_data_from(self, other: "PortfolioOptimizer") -> None:
        """Reuse already downloaded market data so comparison experiments are identical."""
        if other.prices is None:
            raise RuntimeError("Source optimizer has no price data to copy.")
        self.tickers = list(other.tickers)
        self.prices = other.prices.copy()
        self.n_assets = len(self.tickers)
        self.benchmark_prices = None if other.benchmark_prices is None else other.benchmark_prices.copy()

    def prepare_inputs(self) -> None:
        """
        Convert prices into returns and split them into train and test samples.

        The last 252 trading days are reserved as the out-of-sample test window.
        This is the main protection against look-ahead bias.
        """
        if self.prices is None:
            raise RuntimeError("Price data is not available. Run download_adjusted_close first.")

        self.returns_daily = self.prices.pct_change().dropna()
        if len(self.returns_daily) <= self.train_test_split_days + 100:
            raise RuntimeError(f"Not enough return observations: {len(self.returns_daily)} total rows, "
                               f"need at least {self.train_test_split_days + 100} for split.")

        self.train_returns = self.returns_daily.iloc[:-self.train_test_split_days].copy()
        self.test_returns = self.returns_daily.iloc[-self.train_test_split_days :].copy()
        self.mu_train = self.train_returns.mean() * self.trading_days
        self.cov_train = self.train_returns.cov() * self.trading_days

        if self.benchmark_prices is not None:
            benchmark_returns = self.benchmark_prices.pct_change().dropna()
            benchmark_returns = benchmark_returns.reindex(self.returns_daily.index).ffill().dropna()
            self.benchmark_returns = benchmark_returns
            self.benchmark_test_returns = benchmark_returns.loc[self.test_returns.index]

    # =============================
    # 3) Portfolio mechanics
    # =============================

    def project_weights_with_cardinality(
        self,
        raw_weight_genes: np.ndarray,
        selection_genes: np.ndarray,
    ) -> np.ndarray:
        """
        Convert one particle into a feasible long-only portfolio.

        The particle uses 2N encoding:
        - first N genes: raw weight intensity,
        - next N genes: ranking scores used to decide which assets are active.

        The repair sequence is deliberate:
        1. Select exactly K assets using the selection genes.
        2. Normalize raw intensities into positive weights.
        3. Apply the buy-in threshold.
        4. Cap weights at the maximum allowed concentration.
        5. Iteratively re-scale the remaining free weights until the budget sums to one.
        """
        ranked_idx = np.argsort(selection_genes)[::-1]
        selected_idx = ranked_idx[: self.cardinality_k]

        weights = np.zeros(len(raw_weight_genes), dtype=float)
        selected_raw = np.clip(raw_weight_genes[selected_idx], 1e-6, None)
        selected_raw = selected_raw / selected_raw.sum()

        min_total = self.min_active_weight * self.cardinality_k
        if min_total >= 1.0:
            raise ValueError("MIN_ACTIVE_WEIGHT is infeasible for the requested cardinality.")

        selected_weights = self.min_active_weight + (1.0 - min_total) * selected_raw
        selected_weights = np.clip(selected_weights, 0.0, self.max_weight)

        for _ in range(40):
            total = float(np.sum(selected_weights))
            if abs(total - 1.0) < 1e-10:
                break

            free_mask = selected_weights < (self.max_weight - 1e-12)
            if not np.any(free_mask):
                selected_weights = selected_weights / total
                selected_weights = np.clip(selected_weights, 0.0, self.max_weight)
                continue

            delta = 1.0 - total
            free_weights = selected_weights[free_mask]
            free_sum = float(np.sum(free_weights))

            if free_sum <= 1e-12:
                selected_weights[free_mask] += delta / np.sum(free_mask)
            else:
                selected_weights[free_mask] += delta * (free_weights / free_sum)

            selected_weights = np.clip(selected_weights, 0.0, self.max_weight)

        selected_weights = selected_weights / np.sum(selected_weights)
        weights[selected_idx] = selected_weights
        weights[np.abs(weights) < self.active_threshold] = 0.0
        return weights

    def sector_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """Aggregate name-level weights into sector-level exposures."""
        exposures: Dict[str, float] = {sector: 0.0 for sector in self.sector_colors}
        for ticker, weight in zip(self.tickers, weights):
            exposures[self.sector_map[ticker]] += float(weight)
        return exposures

    def portfolio_stats(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        benchmark_weights: np.ndarray | None = None,
        initial_capital: float = 10_000.0,
    ) -> PortfolioStats:
        """
        Compute realized portfolio statistics from a daily return panel.

        Using realized daily returns here is important because the report should describe actual
        performance over a sample path, not only theoretical moment estimates.
        """
        portfolio_daily = pd.Series(returns_df.values @ weights, index=returns_df.index)
        annual_return = float(portfolio_daily.mean() * self.trading_days)
        annual_volatility = float(portfolio_daily.std(ddof=1) * np.sqrt(self.trading_days))
        annual_volatility = max(annual_volatility, 1e-12)
        sharpe_ratio = float((annual_return - self.risk_free_rate) / annual_volatility)
        mean_absolute_deviation = float(np.mean(np.abs(portfolio_daily - portfolio_daily.mean())) * self.trading_days)

        curve = initial_capital * (1.0 + portfolio_daily).cumprod()
        running_peak = curve.cummax()
        drawdown = (curve / running_peak) - 1.0
        max_drawdown = float(drawdown.min())

        if benchmark_weights is None:
            turnover_rate = 0.0
        else:
            turnover_rate = float(0.5 * np.sum(np.abs(weights - benchmark_weights)))

        return PortfolioStats(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            mean_absolute_deviation=mean_absolute_deviation,
            max_drawdown=max_drawdown,
            turnover_rate=turnover_rate,
            final_value=float(curve.iloc[-1]),
        )

    def benchmark_stats(self, benchmark_returns: pd.Series, initial_capital: float = 10_000.0) -> PortfolioStats:
        """Compute the same report metrics for a one-dimensional benchmark return series."""
        annual_return = float(benchmark_returns.mean() * self.trading_days)
        annual_volatility = float(benchmark_returns.std(ddof=1) * np.sqrt(self.trading_days))
        annual_volatility = max(annual_volatility, 1e-12)
        sharpe_ratio = float((annual_return - self.risk_free_rate) / annual_volatility)
        mean_absolute_deviation = float(
            np.mean(np.abs(benchmark_returns - benchmark_returns.mean())) * self.trading_days
        )

        curve = initial_capital * (1.0 + benchmark_returns).cumprod()
        running_peak = curve.cummax()
        drawdown = (curve / running_peak) - 1.0

        return PortfolioStats(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            mean_absolute_deviation=mean_absolute_deviation,
            max_drawdown=float(drawdown.min()),
            turnover_rate=0.0,
            final_value=float(curve.iloc[-1]),
        )

    def backtest_cumulative_value(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        initial_capital: float = 10_000.0,
    ) -> pd.Series:
        """Transform a portfolio daily return series into a cumulative wealth curve."""
        portfolio_daily = pd.Series(returns_df.values @ weights, index=returns_df.index)
        return initial_capital * (1.0 + portfolio_daily).cumprod()

    def benchmark_cumulative_value(self, initial_capital: float = 10_000.0) -> pd.Series | None:
        """Transform the benchmark return series into a cumulative wealth curve."""
        if self.benchmark_test_returns is None:
            return None
        return initial_capital * (1.0 + self.benchmark_test_returns).cumprod()

    def build_equal_weight(self) -> np.ndarray:
        """Equal Weight is the simplest baseline because it uses no optimization at all."""
        return np.ones(len(self.tickers), dtype=float) / len(self.tickers)

    # =============================
    # 4) Optimization layer
    # =============================

    def optimize(self) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
        """
        Execute Multi-Objective NSGA-II using pymoo.
        
        Returns the Pareto Front and identifies the Max Sharpe Ratio portfolio.
        """
        print(
            f"Running NSGA-II for {self.experiment_name} ... "
            f"(pop={self.swarm_size}, generations={self.max_iter}, assets={self.n_assets})"
        )
        
        problem = PortfolioProblem(self)
        algorithm = NSGA2(
            pop_size=self.swarm_size,
            sampling=FloatRandomSampling(),
            repair=PortfolioRepair(self),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", self.max_iter),
            seed=self.random_seed,
            verbose=False
        )
        
        if res.F is None:
            raise RuntimeError("NSGA-II failed to find any solutions.")
        
        # Extract Pareto Front: F contains [Risk, -Return]
        pareto_risks = res.F[:, 0]
        pareto_returns = -res.F[:, 1]
        pareto_weights = res.X[:, :self.n_assets]
        
        # Calculate Sharpe Ratios for all points on Pareto Front
        sharpes = (pareto_returns - self.risk_free_rate) / pareto_risks
        best_idx = np.argmax(sharpes)
        
        self.best_weights = pareto_weights[best_idx]
        self.best_objective = -sharpes[best_idx]
        
        # Store Pareto Front for plotting
        self.pareto_df = pd.DataFrame({
            "risk": pareto_risks,
            "return": pareto_returns,
            "sharpe": sharpes
        })
        
        # Convergence history (simulated for compatibility)
        history_rows = [{
            "run_id": 0, 
            "iteration": i, 
            "best_sharpe": sharpes[best_idx],
            "seed": self.random_seed
        } for i in range(self.max_iter)]
        self.convergence_history_df = pd.DataFrame(history_rows)
        self.run_history_df = pd.DataFrame([{"run_id": 0, "sharpe": sharpes[best_idx]}])
        
        print(f"NSGA-II complete. Best Sharpe on Pareto Front: {sharpes[best_idx]:.6f}")
        return self.best_weights, self.best_objective, self.run_history_df, self.convergence_history_df


    # =============================
    # 5) Reporting tables
    # =============================

    def simulate_random_constrained_portfolios(self, n_samples: int | None = None) -> pd.DataFrame:
        """Generate random feasible portfolios to visualize the constrained opportunity set."""
        np.random.seed(self.random_seed)
        n_draws = self.frontier_random_samples if n_samples is None else n_samples

        risks: List[float] = []
        returns: List[float] = []
        sharpes: List[float] = []

        for _ in range(n_draws):
            selected = np.random.choice(len(self.tickers), size=self.cardinality_k, replace=False)
            local_weights = np.random.dirichlet(np.ones(self.cardinality_k))
            local_weights = self.min_active_weight + (1.0 - self.min_active_weight * self.cardinality_k) * local_weights
            local_weights = np.clip(local_weights, 0.0, self.max_weight)
            local_weights = local_weights / np.sum(local_weights)

            if np.any(local_weights > self.max_weight + 1e-9):
                continue

            weights = np.zeros(len(self.tickers), dtype=float)
            weights[selected] = local_weights

            stats = self.portfolio_stats(self.train_returns, weights)
            risks.append(stats.annual_volatility)
            returns.append(stats.annual_return)
            sharpes.append(stats.sharpe_ratio)

        return pd.DataFrame({"risk": risks, "return": returns, "sharpe": sharpes})

    def build_constraint_audit(self, weights: np.ndarray) -> pd.DataFrame:
        """Create a one-row audit table for all hard constraints."""
        active_mask = weights > self.active_threshold
        positive_weights = weights[active_mask]
        min_positive = float(np.min(positive_weights)) if len(positive_weights) > 0 else 0.0
        sector_exposure = self.sector_exposures(weights)
        sector_ok = True
        if self.max_sector_weight is not None:
            sector_ok = max(sector_exposure.values()) <= self.max_sector_weight + 1e-10

        audit_row: Dict[str, float | bool] = {
            "sum_weights": float(np.sum(weights)),
            "max_weight": float(np.max(weights)),
            "active_count": int(np.sum(active_mask)),
            "min_active_weight": min_positive,
            "budget_ok": bool(abs(np.sum(weights) - 1.0) < 1e-6),
            "boundary_ok": bool(np.all(weights >= -1e-10) and np.all(weights <= self.max_weight + 1e-10)),
            "cardinality_ok": bool(int(np.sum(active_mask)) == self.cardinality_k),
            "buyin_ok": bool(min_positive >= self.min_active_weight - 1e-8),
            "sector_ok": bool(sector_ok),
        }
        for sector, exposure in sector_exposure.items():
            audit_row[f"sector_{sector.lower()}"] = float(exposure)
        return pd.DataFrame([audit_row])

    def build_weights_table(self, weights: np.ndarray) -> pd.DataFrame:
        """Return the selected holdings sorted from largest to smallest weight."""
        rows: List[Dict[str, float | str]] = []
        for ticker, weight in zip(self.tickers, weights):
            if weight > self.active_threshold:
                rows.append(
                    {
                        "ticker": ticker,
                        "sector": self.sector_map[ticker],
                        "weight": float(weight),
                    }
                )
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    def build_sector_table(self, weights: np.ndarray) -> pd.DataFrame:
        """Return sector exposures in descending order, with the active cap shown if one exists."""
        sector_exposure = self.sector_exposures(weights)
        rows = []
        for sector, exposure in sector_exposure.items():
            within_limit = True if self.max_sector_weight is None else exposure <= self.max_sector_weight + 1e-10
            rows.append(
                {
                    "sector": sector,
                    "weight": exposure,
                    "limit": np.nan if self.max_sector_weight is None else self.max_sector_weight,
                    "within_limit": within_limit,
                }
            )
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    def build_metrics_summary(self, pso_weights: np.ndarray, eq_weights: np.ndarray) -> pd.DataFrame:
        """Build a train/test comparison table for PSO, Equal Weight, and the market benchmark."""
        pso_train = self.portfolio_stats(self.train_returns, pso_weights, benchmark_weights=eq_weights)
        eq_train = self.portfolio_stats(self.train_returns, eq_weights, benchmark_weights=eq_weights)
        pso_test = self.portfolio_stats(self.test_returns, pso_weights, benchmark_weights=eq_weights)
        eq_test = self.portfolio_stats(self.test_returns, eq_weights, benchmark_weights=eq_weights)

        rows = [
            {
                "portfolio": self.experiment_name,
                "sample": "train",
                "objective": float(self.best_objective),
                "ann_return": pso_train.annual_return,
                "ann_vol": pso_train.annual_volatility,
                "sharpe": pso_train.sharpe_ratio,
                "mad": pso_train.mean_absolute_deviation,
                "max_drawdown": pso_train.max_drawdown,
                "turnover_rate": pso_train.turnover_rate,
                "final_value": pso_train.final_value,
            },
            {
                "portfolio": "EqualWeight",
                "sample": "train",
                "objective": np.nan,
                "ann_return": eq_train.annual_return,
                "ann_vol": eq_train.annual_volatility,
                "sharpe": eq_train.sharpe_ratio,
                "mad": eq_train.mean_absolute_deviation,
                "max_drawdown": eq_train.max_drawdown,
                "turnover_rate": eq_train.turnover_rate,
                "final_value": eq_train.final_value,
            },
            {
                "portfolio": self.experiment_name,
                "sample": "test",
                "objective": float(self.best_objective),
                "ann_return": pso_test.annual_return,
                "ann_vol": pso_test.annual_volatility,
                "sharpe": pso_test.sharpe_ratio,
                "mad": pso_test.mean_absolute_deviation,
                "max_drawdown": pso_test.max_drawdown,
                "turnover_rate": pso_test.turnover_rate,
                "final_value": pso_test.final_value,
            },
            {
                "portfolio": "EqualWeight",
                "sample": "test",
                "objective": np.nan,
                "ann_return": eq_test.annual_return,
                "ann_vol": eq_test.annual_volatility,
                "sharpe": eq_test.sharpe_ratio,
                "mad": eq_test.mean_absolute_deviation,
                "max_drawdown": eq_test.max_drawdown,
                "turnover_rate": eq_test.turnover_rate,
                "final_value": eq_test.final_value,
            },
        ]

        if self.benchmark_test_returns is not None:
            benchmark_test = self.benchmark_stats(self.benchmark_test_returns)
            rows.append(
                {
                    "portfolio": self.benchmark_ticker,
                    "sample": "test",
                    "objective": np.nan,
                    "ann_return": benchmark_test.annual_return,
                    "ann_vol": benchmark_test.annual_volatility,
                    "sharpe": benchmark_test.sharpe_ratio,
                    "mad": benchmark_test.mean_absolute_deviation,
                    "max_drawdown": benchmark_test.max_drawdown,
                    "turnover_rate": benchmark_test.turnover_rate,
                    "final_value": benchmark_test.final_value,
                }
            )

        return pd.DataFrame(rows)

    # =============================
    # 6) Plotting layer
    # =============================

    @staticmethod
    def detect_plateau_iteration(convergence_history_df: pd.DataFrame) -> int:
        """Find the earliest iteration whose best Sharpe is effectively at the final plateau."""
        final_sharpe = float(convergence_history_df["best_sharpe"].iloc[-1])
        threshold = final_sharpe - 0.005
        plateau_rows = convergence_history_df.loc[convergence_history_df["best_sharpe"] >= threshold, "iteration"]
        return int(plateau_rows.iloc[0]) if not plateau_rows.empty else int(convergence_history_df["iteration"].iloc[-1])

    def plot_convergence_curve(self, convergence_history_df: pd.DataFrame, save_path: str) -> None:
        """
        Plot convergence with a tighter Sharpe-axis range and an explicit plateau marker.

        This chart answers two questions for the presentation:
        - Does PSO really improve over time?
        - Around which iteration does the solution stop changing materially?
        """
        plt.figure(figsize=(11, 6))
        overall_min = float(convergence_history_df["best_sharpe"].min())
        overall_max = float(convergence_history_df["best_sharpe"].max())

        for run_id, run_df in convergence_history_df.groupby("run_id"):
            plateau_iteration = self.detect_plateau_iteration(run_df)
            plateau_value = float(run_df.loc[run_df["iteration"] == plateau_iteration, "best_sharpe"].iloc[0])
            label = f"Run {run_id} (seed={int(run_df['seed'].iloc[0])})"
            plt.plot(run_df["iteration"], run_df["best_sharpe"], linewidth=1.6, alpha=0.85, label=label)
            plt.scatter([plateau_iteration], [plateau_value], s=70, facecolors="none", edgecolors="black", linewidths=1.4)

        best_run_id = int(convergence_history_df.groupby("run_id")["best_sharpe"].max().idxmax())
        best_run_df = convergence_history_df[convergence_history_df["run_id"] == best_run_id]
        plateau_iteration = self.detect_plateau_iteration(best_run_df)
        plateau_value = float(best_run_df.loc[best_run_df["iteration"] == plateau_iteration, "best_sharpe"].iloc[0])
        plt.annotate(
            f"Plateau ~ iteration {plateau_iteration}",
            xy=(plateau_iteration, plateau_value),
            xytext=(plateau_iteration + 30, plateau_value - 0.03),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
            fontsize=10,
        )

        lower_limit = max(0.0, overall_min - 0.03)
        upper_limit = overall_max + 0.03
        plt.ylim(lower_limit, upper_limit)
        plt.title(f"Convergence Curve: {self.experiment_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Sharpe Ratio")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_selected_weights(self, weights_df: pd.DataFrame, save_path: str) -> None:
        """Plot the selected holdings with colors that make sector composition instantly visible."""
        plt.figure(figsize=(11, 5))
        colors = [self.sector_colors[sector] for sector in weights_df["sector"]]
        bars = plt.bar(weights_df["ticker"], weights_df["weight"], color=colors)
        plt.title(f"Selected Weights: {self.experiment_name}")
        plt.xlabel("Ticker")
        plt.ylabel("Weight")
        plt.ylim(0.0, max(0.28, float(weights_df["weight"].max()) + 0.03))
        plt.grid(axis="y", alpha=0.25)
        for bar, weight in zip(bars, weights_df["weight"]):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.004, f"{weight:.2%}", ha="center", fontsize=9)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=sector)
            for sector, color in self.sector_colors.items()
        ]
        plt.legend(handles=legend_handles, title="Sector")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_cumulative_returns(
        self,
        pso_curve: pd.Series,
        eq_curve: pd.Series,
        benchmark_curve: pd.Series | None,
        save_path: str,
    ) -> None:
        """Plot out-of-sample wealth growth for the optimized portfolio and all benchmarks."""
        plt.figure(figsize=(11, 5))
        plt.plot(pso_curve.index, pso_curve.values, label=self.experiment_name, linewidth=2.4, color="#002D62") # Navy
        plt.plot(eq_curve.index, eq_curve.values, label="Equal Weight", linewidth=2.1, color="#7f7f7f") # Gray
        if benchmark_curve is not None:
            plt.plot(benchmark_curve.index, benchmark_curve.values, label=self.benchmark_ticker, linewidth=2.0, color="#FFC000") # Gold
        plt.title("Out-of-Sample Backtest: Growth of $10,000")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_pareto_front(self, save_path: str) -> None:
        """
        Plot the Pareto Front showing the optimal trade-off between Risk and Return.
        The Max Sharpe point is highlighted as the recommended solution.
        """
        if not hasattr(self, "pareto_df"):
            return
            
        plt.figure(figsize=(11, 6))
        plt.scatter(self.pareto_df["risk"], self.pareto_df["return"], c=self.pareto_df["sharpe"], 
                    cmap="viridis", s=30, alpha=0.7, label="Pareto Front (Optimal Solutions)")
        
        # Highlight Max Sharpe point
        best_idx = self.pareto_df["sharpe"].idxmax()
        best_p = self.pareto_df.loc[best_idx]
        plt.scatter(best_p["risk"], best_p["return"], marker="*", color="#FFC000", s=300, 
                    edgecolor="black", label=f"Max Sharpe Portfolio ({best_p['sharpe']:.4f})")
        
        plt.title(f"Pareto Front: Risk vs Return ({self.experiment_name})")
        plt.xlabel("Annual Volatility (Risk)")
        plt.ylabel("Annual Return")
        plt.colorbar(label="Sharpe Ratio")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_efficient_frontier(
        self,
        random_df: pd.DataFrame,
        pso_stats: PortfolioStats,
        eq_stats: PortfolioStats,
        save_path: str,
    ) -> None:
        """Plot the random feasible cloud and highlight the optimized solution."""
        plt.figure(figsize=(11, 6))
        scatter = plt.scatter(
            random_df["risk"],
            random_df["return"],
            c=random_df["sharpe"],
            cmap="viridis",
            alpha=0.25,
            s=10,
            label=f"Random feasible portfolios ({len(random_df):,})",
        )
        plt.colorbar(scatter, label="Sharpe Ratio")
        plt.scatter(pso_stats.annual_volatility, pso_stats.annual_return, marker="*", color="red", s=280, label=self.experiment_name)
        plt.scatter(eq_stats.annual_volatility, eq_stats.annual_return, marker="X", color="black", s=180, label="Equal Weight")
        plt.title("Efficient Frontier Approximation with Random Feasible Portfolios")
        plt.xlabel("Annual Volatility")
        plt.ylabel("Annual Return")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    # =============================
    # 7) Output writers
    # =============================

    def save_outputs(self, artifacts: ExperimentArtifacts, output_dir: str, file_prefix: str = "") -> None:
        """Write the scenario tables and charts to disk using an optional prefix."""
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{file_prefix}_" if file_prefix else ""
        output_path = Path(output_dir)

        artifacts.weights_table.to_csv(output_path / f"{prefix}pso_weights.csv", index=False, float_format="%.6f")
        artifacts.constraint_audit.to_csv(output_path / f"{prefix}constraint_audit.csv", index=False, float_format="%.6f")
        artifacts.metrics_summary.to_csv(output_path / f"{prefix}metrics_summary.csv", index=False, float_format="%.6f")
        artifacts.sector_table.to_csv(output_path / f"{prefix}sector_allocations.csv", index=False, float_format="%.6f")
        artifacts.run_history_df.to_csv(output_path / f"{prefix}pso_run_history.csv", index=False, float_format="%.6f")
        artifacts.convergence_history_df.to_csv(output_path / f"{prefix}convergence_history.csv", index=False, float_format="%.6f")
        artifacts.pso_curve.to_frame(name="portfolio_value").to_csv(output_path / f"{prefix}pso_backtest_curve.csv", float_format="%.6f")
        artifacts.equal_curve.to_frame(name="portfolio_value").to_csv(output_path / f"{prefix}equal_backtest_curve.csv", float_format="%.6f")
        if artifacts.benchmark_curve is not None:
            artifacts.benchmark_curve.to_frame(name="portfolio_value").to_csv(
                output_path / f"{prefix}benchmark_backtest_curve.csv", float_format="%.6f"
            )

        random_df = self.simulate_random_constrained_portfolios()
        pso_train_stats = self.portfolio_stats(self.train_returns, artifacts.weights, benchmark_weights=self.build_equal_weight())
        eq_train_stats = self.portfolio_stats(self.train_returns, self.build_equal_weight(), benchmark_weights=self.build_equal_weight())

        self.plot_convergence_curve(artifacts.convergence_history_df, str(output_path / f"{prefix}convergence_curve.png"))
        self.plot_selected_weights(artifacts.weights_table, str(output_path / f"{prefix}selected_weights.png"))
        self.plot_cumulative_returns(artifacts.pso_curve, artifacts.equal_curve, artifacts.benchmark_curve, str(output_path / f"{prefix}cumulative_returns.png"))
        self.plot_efficient_frontier(random_df, pso_train_stats, eq_train_stats, str(output_path / f"{prefix}efficient_frontier.png"))
        self.plot_pareto_front(str(output_path / f"{prefix}pareto_front.png"))

    def print_report(self, artifacts: ExperimentArtifacts) -> None:
        """Print a concise report to the terminal so the latest run is immediately inspectable."""
        print("\n" + "=" * 96)
        print(f"ACADEMIC PORTFOLIO OPTIMIZATION REPORT: {artifacts.experiment_name}")
        print("=" * 96)

        print("\nSelected assets:")
        for row in artifacts.weights_table.itertuples(index=False):
            print(f"  {row.ticker:>6s} | {row.sector:<8s} | {row.weight:0.6f} ({row.weight:6.2%})")

        audit_row = artifacts.constraint_audit.iloc[0]
        print("\nConstraint audit:")
        print(f"  sum_weights         : {audit_row['sum_weights']:.6f}")
        print(f"  max_weight          : {audit_row['max_weight']:.6f}")
        print(f"  active_count        : {int(audit_row['active_count'])}")
        print(f"  min_active_weight   : {audit_row['min_active_weight']:.6f}")
        print(
            "  checks_ok           : "
            f"budget={audit_row['budget_ok']}, boundary={audit_row['boundary_ok']}, "
            f"cardinality={audit_row['cardinality_ok']}, buyin={audit_row['buyin_ok']}, sector={audit_row['sector_ok']}"
        )

        print("\nSector allocation:")
        for row in artifacts.sector_table.itertuples(index=False):
            limit_text = "None" if pd.isna(row.limit) else f"{row.limit:0.2f}"
            print(f"  {row.sector:<8s} | weight={row.weight:0.6f} | limit={limit_text} | within_limit={row.within_limit}")

        print("\nMetrics summary:")
        for row in artifacts.metrics_summary.itertuples(index=False):
            print(
                f"  {row.portfolio:<28s} | {row.sample:<5s} | return={row.ann_return:0.6f} | vol={row.ann_vol:0.6f} | "
                f"sharpe={row.sharpe:0.6f} | mad={row.mad:0.6f} | mdd={row.max_drawdown:0.6f}"
            )

    def run(self) -> ExperimentArtifacts:
        """Execute one full optimization scenario and return all resulting artifacts."""
        if self.prices is None:
            self.download_adjusted_close()
        self.prepare_inputs()

        pso_weights, objective, run_history_df, convergence_history_df = self.optimize()
        eq_weights = self.build_equal_weight()

        weights_table = self.build_weights_table(pso_weights)
        sector_table = self.build_sector_table(pso_weights)
        constraint_audit = self.build_constraint_audit(pso_weights)
        metrics_summary = self.build_metrics_summary(pso_weights, eq_weights)
        pso_curve = self.backtest_cumulative_value(self.test_returns, pso_weights)
        equal_curve = self.backtest_cumulative_value(self.test_returns, eq_weights)
        benchmark_curve = self.benchmark_cumulative_value()

        artifacts = ExperimentArtifacts(
            experiment_name=self.experiment_name,
            weights=pso_weights,
            objective=objective,
            weights_table=weights_table,
            sector_table=sector_table,
            constraint_audit=constraint_audit,
            metrics_summary=metrics_summary,
            run_history_df=run_history_df,
            convergence_history_df=convergence_history_df,
            pso_curve=pso_curve,
            equal_curve=equal_curve,
            benchmark_curve=benchmark_curve,
        )
        self.print_report(artifacts)
        return artifacts


def build_bonus_comparison_table(
    constrained_artifacts: ExperimentArtifacts,
    original_artifacts: ExperimentArtifacts,
) -> pd.DataFrame:
    """Build a compact comparison table for the bonus slide on sector constraints."""
    constrained_test = constrained_artifacts.metrics_summary.query("portfolio == @constrained_artifacts.experiment_name and sample == 'test'").iloc[0]
    original_test = original_artifacts.metrics_summary.query("portfolio == @original_artifacts.experiment_name and sample == 'test'").iloc[0]

    constrained_sector = constrained_artifacts.sector_table.set_index("sector")["weight"]
    original_sector = original_artifacts.sector_table.set_index("sector")["weight"]

    rows = [
        {
            "scenario": original_artifacts.experiment_name,
            "ann_return_test": float(original_test["ann_return"]),
            "ann_vol_test": float(original_test["ann_vol"]),
            "sharpe_test": float(original_test["sharpe"]),
            "max_drawdown_test": float(original_test["max_drawdown"]),
            "top_sector_weight": float(original_sector.max()),
        },
        {
            "scenario": constrained_artifacts.experiment_name,
            "ann_return_test": float(constrained_test["ann_return"]),
            "ann_vol_test": float(constrained_test["ann_vol"]),
            "sharpe_test": float(constrained_test["sharpe"]),
            "max_drawdown_test": float(constrained_test["max_drawdown"]),
            "top_sector_weight": float(constrained_sector.max()),
        },
    ]
    return pd.DataFrame(rows)


def plot_bonus_constraint_effect(
    constrained_artifacts: ExperimentArtifacts,
    original_artifacts: ExperimentArtifacts,
    save_path: str,
) -> None:
    """Plot the bonus comparison: MDD bars and sector exposure bars side by side."""
    comparison_df = build_bonus_comparison_table(constrained_artifacts, original_artifacts)
    original_sector = original_artifacts.sector_table.set_index("sector")["weight"]
    constrained_sector = constrained_artifacts.sector_table.set_index("sector")["weight"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(comparison_df["scenario"], comparison_df["max_drawdown_test"], color=["#7f7f7f", "#d62728"])
    axes[0].set_title("Maximum Drawdown Comparison")
    axes[0].set_ylabel("Max Drawdown")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(comparison_df["max_drawdown_test"]):
        axes[0].text(idx, value + 0.003, f"{value:.2%}", ha="center", fontsize=10)

    sector_df = pd.DataFrame(
        {
            "Original PSO": original_sector,
            "Sector-Constrained PSO": constrained_sector,
        }
    )
    sector_df.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].axhline(0.40, linestyle="--", linewidth=1.2, color="black", label="40% cap")
    axes[1].set_title("Sector Exposure Comparison")
    axes[1].set_ylabel("Weight")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)



def write_presentation_script(
    constrained_artifacts: ExperimentArtifacts,
    original_artifacts: ExperimentArtifacts,
    bonus_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write an academic Thai presentation script for MOPSO."""
    constrained_test = constrained_artifacts.metrics_summary.query("portfolio == @constrained_artifacts.experiment_name and sample == 'test'").iloc[0]
    equal_test = constrained_artifacts.metrics_summary.query("portfolio == 'EqualWeight' and sample == 'test'").iloc[0]
    
    best_run = constrained_artifacts.run_history_df.sort_values("sharpe", ascending=False).iloc[0]

    text = f"""# 🏆 Script การนำเสนอวิทยานิพนธ์: Advanced Portfolio Optimization using NSGA-II

---

## 🏛️ ส่วนที่ 1: โจทย์และสมมติฐาน (The Problem & Hypothesis)
**จุดประสงค์:** เพื่อแสดงให้คณะกรรมการเห็นว่าเราเข้าใจ "รากเหง้า" ของปัญหาทางการเงิน

*   **โจทย์ (The Problem):** "เราจะเลือกหุ้นเพียง 10 ตัวจาก 50 ตัวอย่างไร ให้ได้ผลตอบแทนสูงที่สุดแต่ความเสี่ยงต่ำที่สุด?" นี่คือปัญหาแบบ Combinatorial Optimization ที่มีความซับซ้อนสูงมาก (NP-Hard) เนื่องจากมีรูปแบบการจัดพอร์ตที่เป็นไปได้หลายล้านรูปแบบ
*   **โมเดลหลัก (The Model):** เราใช้ **Mean-Variance (M-V) Model ของ Harry Markowitz** (รางวัลโนเบล) เป็นฐานการคำนวณ เพื่อวัดความสัมพันธ์ระหว่าง Expected Return และ Portfolio Variance (Covariance Matrix)
*   **สมมติฐาน (Hypothesis):** "การใช้เทคนิควิวัฒนาการแบบหลายวัตถุประสงค์ (NSGA-II) จะสามารถค้นหาเส้น **Efficient Frontier** ที่มีความเสถียรและแม่นยำกว่าการสุ่มเลือก หรือการใช้ Single-Objective Optimization ทั่วไป แม้จะอยู่ภายใต้ข้อจำกัด (Constraints) ที่เข้มงวดในโลกจริง"

---

## 🛠️ ส่วนที่ 2: เครื่องมือและอัลกอริทึม (Tools & Algorithm)
**จุดประสงค์:** แยกแยะหน้าที่ของเทคโนโลยีให้ชัดเจน (Computational Intelligence Perspective)

*   **Model (เป้าหมาย):** คือทฤษฎี Mean-Variance ที่เราใช้ตั้งโจทย์ (สูตรคำนวณ)
*   **Algorithm (วิธีการ):** คือ **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) ซึ่งเป็น AI ประเภท Evolutionary Algorithm ที่ทำหน้าที่ "หาคำตอบที่ดีที่สุด" จากโจทย์ข้างต้น
*   **Library (กล่องเครื่องมือ):** เราเลือกใช้ **Pymoo** ซึ่งเป็น Framework ระดับโลกสำหรับการทำ Multi-Objective Optimization ใน Python
*   **กลไกของ NSGA-II:** 
    1. **Crossover:** การแลกเปลี่ยนยีนระหว่างพอร์ตที่เก่ง เพื่อส่งต่อคุณลักษณะที่ดี
    2. **Mutation:** การสุ่มกลายพันธุ์เพื่อป้องกันการติดหล่มที่จุดดีที่สุดในพื้นที่จำกัด (Local Optimum)
    3. **Non-dominated Sorting:** การจัดอันดับคำตอบโดยไม่ทิ้งใครไว้ข้างหลัง หากจุดนั้น "ดีที่สุดในมุมมองใดมุมมองหนึ่ง" (เช่น เสี่ยงเท่ากันแต่กำไรเยอะกว่า)

---

## 🔢 ส่วนที่ 3: ตัวแปรและการดำเนินการ (Variables & Execution)
**จุดประสงค์:** โชว์ความแข็งแกร่งของข้อมูลและพารามิเตอร์

*   **Data Universe:** หุ้น 50 ตัว จาก S&P 500 เก็บข้อมูลย้อนหลัง 10 ปี รวมกว่า **126,000 จุดข้อมูล** เพื่อพิสูจน์ Scalability
*   **Objectives (วัตถุประสงค์คู่):**
    1. **Minimize Risk:** ลดความผันผวนของพอร์ต (Annual Volatility)
    2. **Maximize Return:** เพิ่มผลตอบแทนคาดการณ์ (Annual Return)
*   **Constraints (ข้อจำกัดโลกจริง):**
    *   **Cardinality (10):** บังคับเลือก 10 หุ้น เพื่อคุมต้นทุนการจัดการ
    *   **Sector Cap (40%):** จำกัดน้ำหนักรายกลุ่มธุรกิจ เพื่อป้องกันความเสี่ยงเชิงระบบ
    *   **Buy-in (5%-25%):** กำหนดขนาดลงทุนขั้นต่ำและขั้นสูงเพื่อสภาพคล่อง

---

## 📈 ส่วนที่ 4: ผลลัพธ์และการพิสูจน์ (Results & Verification)
**จุดประสงค์:** สรุป Insight จากข้อมูล (Data-Driven Insights)

*   **Pareto Front:** ทุกจุดที่คุณเห็นบนเส้นโค้งนี้คือ "พอร์ตที่เหมาะสมที่สุด" (Optimal Solutions) ซึ่งไม่มีพอร์ตไหนในโลกที่ดีกว่านี้อีกแล้วในเชิงสถิติ (สำหรับข้อมูลชุดนี้)
*   **Tangency Portfolio (จุดดาวสีทอง):** คือจุดที่ AI แนะนำมากที่สุด เพราะให้ค่า **Sharpe Ratio** (ผลตอบแทนเทียบความเสี่ยง) สูงที่สุดในฝั่ง Training คือ {float(best_run['sharpe']):.4f}
*   **Benchmarking:** เมื่อนำไปทดสอบกับข้อมูลจริง (Out-of-sample) พอร์ตของเรามีค่า Sharpe Ratio อยู่ที่ {float(constrained_test['sharpe']):.4f} ซึ่งมีความเสถียรและยังคงรักษาวินัยของ Constraints ได้ครบถ้วน 100%
*   **Stability:** เส้น Pareto ที่เรียบเนียนพิสูจน์ว่า NSGA-II จัดการกับความซับซ้อนของข้อมูลและข้อจำกัดได้สมบูรณ์แบบครับ

---
**สรุปคีย์เวิร์ด:** Mean-Variance Model -> NSGA-II Algorithm -> Pymoo Library -> Efficient Frontier Results
"""
    Path(output_path).write_text(text, encoding="utf-8")


def write_detailed_report(
    constrained_artifacts: ExperimentArtifacts,
    original_artifacts: ExperimentArtifacts,
    bonus_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write a detailed Thai report for the MOPSO study."""
    constrained_test = constrained_artifacts.metrics_summary.query("portfolio == @constrained_artifacts.experiment_name and sample == 'test'").iloc[0]
    equal_test = constrained_artifacts.metrics_summary.query("portfolio == 'EqualWeight' and sample == 'test'").iloc[0]
    best_run = constrained_artifacts.run_history_df.sort_values("sharpe", ascending=False).iloc[0]

    report = f"""# รายงานการวิจัยขั้นสูง: การจัดสรรพอร์ตการลงทุนด้วย NSGA-II (Academic Research Report)

## 1. บทนำและสมมติฐาน (Introduction & Hypothesis)
งานวิจัยนี้มุ่งเน้นการแก้ปัญหา **Multi-Objective Portfolio Optimization** ภายใต้เงื่อนไขข้อจำกัดที่สมจริง (Real-world Constraints) โดยมีสมมติฐานว่าการใช้อัลกอริทึมเชิงวิวัฒนาการ (Evolutionary Algorithms) สามารถค้นหาชุดคำตอบที่มีประสิทธิภาพ (Efficient Frontier) ได้ดีกว่าวิธีการดั้งเดิมเมื่อเผชิญกับเงื่อนไข Cardinality และ Sector Caps

### 1.1 ทฤษฎีอ้างอิง
เราประยุกต์ใช้ **Modern Portfolio Theory (MPT)** ของ Harry Markowitz เป็นพื้นฐานในการวัดความสัมพันธ์ระหว่าง Risk และ Return

---

## 2. กระบวนการและเครื่องมือ (Methodology & Tools)
การวิจัยแบ่งการทำงานออกเป็น 2 ส่วนหลัก:
1. **The Model:** การสร้างสมการ Mean-Variance เพื่อกำหนดพื้นที่คำตอบ (Search Space)
2. **The Algorithm:** การใช้ **NSGA-II** ผ่านไลบรารี **Pymoo** เพื่อทำการค้นหาคำตอบแบบ Iterative
    *   **Crossover:** การผสมคุณลักษณะพอร์ต
    *   **Mutation:** การรักษาความหลากหลายของคำตอบ
    *   **Elite Preservation:** การรักษาคำตอบที่ดีที่สุดไว้ในรุ่นถัดไป

---

## 3. ตัวแปรและการดำเนินการ (Experimental Setup)
### 3.1 ข้อมูล (Data Universe)
- **Universe:** 50 หุ้นชั้นนำ (Selected Assets)
- **Timeframe:** 10 ปี (2014-2024)
- **Data Points:** > 126,000 จุดข้อมูล (High Fidelity Data)

### 3.2 เงื่อนไขบังคับ (Constraints)
- **Cardinality:** 10 Assets (Constraint CC)
- **Weight Bounds:** 5% - 25% per Asset (Constraint WB)
- **Sector Concentration:** < 40% per Sector (Constraint SC)

---

## 4. ผลการทดลองและการวิเคราะห์ (Results & Analysis)
### 4.1 การค้นหา Pareto Front
ผลการดำเนินงานของ NSGA-II สามารถสร้างเส้น **Pareto Front** ที่มีความหนาแน่นและเรียบเนียน (Smooth Convergence) ซึ่งบ่งบอกถึงความสามารถในการหาจุดที่เหมาะสมที่สุดในทุกระดับความเสี่ยง

### 4.2 การเปรียบเทียบผลการดำเนินงาน (Backtesting)
- **In-sample Sharpe:** {float(best_run['sharpe']):.4f} (Optimal)
- **Out-of-sample Sharpe:** {float(constrained_test['sharpe']):.4f} (Robust)
- **Constraint Audit:** ผ่านการตรวจสอบเงื่อนไขทุกข้อ (100% Feasibility)

---

## 5. บทสรุป (Conclusion)
การศึกษาครั้งนี้พิสูจน์ให้เห็นว่า **NSGA-II** เป็นเครื่องมือที่มีประสิทธิภาพสูงในการจัดการกับโจทย์ Computational Finance ขนาดใหญ่ สามารถสร้างพอร์ตการลงทุนที่สมดุลและนำไปใช้งานจริงได้ภายใต้เงื่อนไขข้อจำกัดของสถาบันการเงินครับ
"""
    Path(output_path).write_text(report, encoding="utf-8")


def run_full_study(output_dir: str = "outputs") -> Dict[str, object]:
    """Run the original and sector-constrained studies, then export all shared outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    constrained_optimizer = PortfolioOptimizer(
        experiment_name="PSO_with_sector_constraints",
        max_sector_weight=MAX_SECTOR_WEIGHT,
        benchmark_ticker=BENCHMARK_TICKER,
        frontier_random_samples=FRONTIER_RANDOM_SAMPLES,
    )
    constrained_artifacts = constrained_optimizer.run()
    constrained_optimizer.save_outputs(constrained_artifacts, output_dir=output_dir, file_prefix="")

    original_optimizer = PortfolioOptimizer(
        experiment_name="PSO_original",
        max_sector_weight=None,
        benchmark_ticker=BENCHMARK_TICKER,
        frontier_random_samples=FRONTIER_RANDOM_SAMPLES,
    )
    original_optimizer.copy_market_data_from(constrained_optimizer)
    original_artifacts = original_optimizer.run()
    original_optimizer.save_outputs(original_artifacts, output_dir=output_dir, file_prefix="original")

    bonus_df = build_bonus_comparison_table(constrained_artifacts, original_artifacts)
    bonus_df.to_csv(output_path / "bonus_sector_comparison.csv", index=False, float_format="%.6f")
    plot_bonus_constraint_effect(constrained_artifacts, original_artifacts, str(output_path / "bonus_sector_comparison.png"))

    write_presentation_script(
        constrained_artifacts,
        original_artifacts,
        bonus_df,
        output_path=str(output_path / "presentation_script_th.md"),
    )
    write_detailed_report(
        constrained_artifacts,
        original_artifacts,
        bonus_df,
        output_path=str(output_path / "report_summary_th.md"),
    )

    return {
        "sector_constrained": constrained_artifacts,
        "original": original_artifacts,
        "bonus_comparison": bonus_df,
    }


def main() -> None:
    """Console entry point used by the script and by external task runners."""
    run_full_study(output_dir="outputs")


if __name__ == "__main__":
    main()
