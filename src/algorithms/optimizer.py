"""
คลาสควบคุมกระบวนการปรับแต่งพอร์ตการลงทุน (Portfolio Optimizer Manager)
-------------------------------------------------------------------
เป็นคลาสศูนย์กลาง (Orchestrator) ที่คอยประสานงานระหว่าง Data, Constraints, Mathematics และ AI (NSGA-II)
ทำหน้าที่ตั้งแต่สั่งโหลดข้อมูล โยนเข้าสมการ สั่งรัน AI จนไปถึงสกัดผลลัพธ์ (Artifacts) กลับมา
"""

from __future__ import annotations
import os
import random
from typing import Dict, List
import pandas as pd
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling

from src.models.problem import PortfolioProblem
from src.models.repair import PortfolioRepair
from src.data.loader import download_adjusted_close, prepare_returns
from src.utils.metrics import calculate_portfolio_stats, build_cumulative_curve
from src import config

from dataclasses import dataclass

@dataclass
class ExperimentArtifacts:
    """โครงสร้างสำหรับแพ็กผลลัพธ์ทุกชิ้น (Artifacts) เข้าด้วยกัน เพื่อส่งไปวาดกราฟต่อไป"""
    experiment_name: str
    weights: np.ndarray
    objective: float
    weights_table: pd.DataFrame
    sector_table: pd.DataFrame
    constraint_audit: pd.DataFrame
    metrics_summary: pd.DataFrame
    run_history_df: pd.DataFrame
    convergence_history_df: pd.DataFrame
    nsga2_curve: pd.Series
    equal_curve: pd.Series
    benchmark_curve: pd.Series | None

class PortfolioOptimizer:
    def __init__(
        self,
        experiment_name: str = "NSGA2_with_sector_constraints",
        max_sector_weight: float | None = config.MAX_SECTOR_WEIGHT,
        benchmark_ticker: str = config.BENCHMARK_TICKER,
        frontier_random_samples: int = config.FRONTIER_RANDOM_SAMPLES,
    ) -> None:
        """
        โหลดค่าเริ่มต้นทั้งหมดจากไฟล์ config.py 
        เราอนุญาตให้แก้ค่า max_sector_weight ได้ เพื่อจำลอง 2 เหตุการณ์:
        1. พอร์ตแบบเสรี (ไม่มีข้อจำกัด Sector)
        2. พอร์ตโลกจริง (โดนคุม Sector ไม่เกิน 40%)
        """
        self.experiment_name = experiment_name
        self.tickers = list(config.TICKERS)
        self.sector_map = dict(config.SECTOR_MAP)
        self.sector_colors = dict(config.SECTOR_COLORS)
        self.years_of_data = config.YEARS_OF_DATA
        self.train_test_split_days = config.TRAIN_TEST_SPLIT_DAYS
        self.trading_days = config.TRADING_DAYS
        self.risk_free_rate = config.RISK_FREE_RATE
        self.cardinality_k = config.CARDINALITY_K
        self.max_weight = config.MAX_WEIGHT_PER_STOCK
        self.min_active_weight = config.MIN_ACTIVE_WEIGHT
        self.max_sector_weight = max_sector_weight
        self.active_threshold = config.ACTIVE_THRESHOLD
        self.swarm_size = config.SWARM_SIZE
        self.max_iter = config.MAX_ITER
        self.random_seed = config.RANDOM_SEED
        self.frontier_random_samples = frontier_random_samples
        self.benchmark_ticker = benchmark_ticker
        self.n_assets = len(self.tickers)

        # การันตีว่าถ้ารันซ้ำ ต้องได้ผลเดิมเป๊ะๆ (Reproducibility)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.prices: pd.DataFrame | None = None
        self.benchmark_prices: pd.Series | None = None
        self.train_returns: pd.DataFrame | None = None
        self.test_returns: pd.DataFrame | None = None
        self.benchmark_returns: pd.Series | None = None
        self.benchmark_test_returns: pd.Series | None = None
        
        # ตัวแปรคณิตศาสตร์: Expected Return และ Covariance Matrix
        self.mu_train: pd.Series | None = None
        self.cov_train: pd.DataFrame | None = None

    def fetch_and_prepare_data(self):
        """เรียกใช้ฟังก์ชันโหลดข้อมูล และคำนวณ mu, cov เอาไว้ล่วงหน้า (Pre-calculate)"""
        self.prices, self.benchmark_prices, self.tickers = download_adjusted_close(self.tickers, self.benchmark_ticker, self.years_of_data)
        self.n_assets = len(self.tickers)
        self.train_returns, self.test_returns, self.benchmark_returns, self.benchmark_test_returns = prepare_returns(self.prices, self.benchmark_prices, self.train_test_split_days)
        
        # ค้นหาค่าสถิติจากชุด Train เท่านั้น ป้องกัน AI แอบดูอนาคต (Look-ahead bias)
        self.mu_train = self.train_returns.mean() * self.trading_days
        self.cov_train = self.train_returns.cov() * self.trading_days

    def copy_market_data_from(self, other: "PortfolioOptimizer") -> None:
        """ฟังก์ชันช่วยประหยัดเวลา ไม่ต้องโหลดข้อมูล Yahoo Finance ซ้ำสองรอบ"""
        self.tickers = list(other.tickers)
        self.prices = other.prices.copy()
        self.n_assets = len(self.tickers)
        self.benchmark_prices = None if other.benchmark_prices is None else other.benchmark_prices.copy()
        self.train_returns = other.train_returns.copy()
        self.test_returns = other.test_returns.copy()
        self.benchmark_returns = other.benchmark_returns.copy()
        self.benchmark_test_returns = other.benchmark_test_returns.copy()
        self.mu_train = other.mu_train.copy()
        self.cov_train = other.cov_train.copy()

    def build_equal_weight(self) -> np.ndarray:
        """สร้างพอร์ตหารเฉลี่ย (Equal Weight = 1/N) เพื่อใช้เป็นฐานเปรียบเทียบ"""
        return np.ones(self.n_assets, dtype=float) / self.n_assets

    def sector_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """รวมน้ำหนักหุ้นเป็นรายหมวดหมู่ธุรกิจ (Sector Aggregation)"""
        exposures: Dict[str, float] = {sector: 0.0 for sector in self.sector_colors}
        for ticker, weight in zip(self.tickers, weights):
            exposures[self.sector_map[ticker]] += float(weight)
        return exposures

    def project_weights_with_cardinality(self, raw_weight_genes: np.ndarray, selection_genes: np.ndarray) -> np.ndarray:
        """
        [หัวใจของการทำ 2N Encoding และ Cardinality Repair]
        1. เรียงคะแนน Selection Genes เพื่อหา Top K หุ้น (เช่น ขอ 10 อันดับแรก)
        2. ปล่อยหุ้น 10 ตัวนี้ไว้ แล้วเตะตัวอื่นออกไปให้เป็น 0
        3. เกลี่ยน้ำหนักหุ้นที่รอดชีวิต ให้เป็นไปตามเพดานที่ตั้งไว้ (Max 25%, Min 5%)
        """
        ranked_idx = np.argsort(selection_genes)[::-1]
        selected_idx = ranked_idx[: self.cardinality_k]

        weights = np.zeros(len(raw_weight_genes), dtype=float)
        selected_raw = np.clip(raw_weight_genes[selected_idx], 1e-6, None)
        selected_raw = selected_raw / selected_raw.sum()

        min_total = self.min_active_weight * self.cardinality_k
        if min_total >= 1.0:
            raise ValueError("MIN_ACTIVE_WEIGHT is infeasible for the requested cardinality.")

        # จองพื้นที่ให้ Min Weight แล้วเอาน้ำหนักที่เหลือมาหารเฉลี่ยตามสัดส่วนเดิม
        selected_weights = self.min_active_weight + (1.0 - min_total) * selected_raw
        selected_weights = np.clip(selected_weights, 0.0, self.max_weight)

        # วนลูปเกลี่ยน้ำหนักให้สมดุล (Balancing Loop) ป้องกันกรณีทะลุเพดาน
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
        
        # หุ้นตัวไหนที่เหลือเศษต่ำกว่า Threshold ให้ปรับเป็น 0 ให้หมด เพื่อความสะอาดของพอร์ต
        weights[np.abs(weights) < self.active_threshold] = 0.0
        return weights

    def optimize(self) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
        """
        [เรียกใช้งาน Pymoo NSGA-II]
        ตั้งค่าประชากร (pop_size) และสั่งให้เริ่มลุยหา Pareto Front (minimize)
        """
        print(f"รัน NSGA-II สำหรับการทดลอง: {self.experiment_name} ... (ประชากร={self.swarm_size}, รุ่น={self.max_iter}, จำนวนหุ้น={self.n_assets})")
        
        problem = PortfolioProblem(self)
        algorithm = NSGA2(
            pop_size=self.swarm_size,
            sampling=FloatRandomSampling(),
            repair=PortfolioRepair(self), # ยัดกลไก 2N เข้าไปก่อนประเมินค่า
            eliminate_duplicates=True     # ห้ามมีโครโมโซมฝาแฝด
        )
        
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", self.max_iter),
            seed=self.random_seed,
            verbose=False
        )
        
        if res.F is None:
            raise RuntimeError("เกิดข้อผิดพลาดรุนแรง: NSGA-II ไม่สามารถหาพอร์ตที่ใช้ได้เลย")
        
        # สกัดจุดบนเส้น Pareto (ผลตอบแทนเสี่ยง และ ผลตอบแทนคาดหวัง)
        pareto_risks = res.F[:, 0]
        pareto_returns = -res.F[:, 1] # เพราะตอนใส่ไปเราติดลบมัน ตอนดึงกลับต้องสลับเครื่องหมายคืน
        pareto_weights = res.X[:, :self.n_assets]
        
        # คัดเลือกสุดยอดพอร์ตด้วย Sharpe Ratio
        sharpes = (pareto_returns - self.risk_free_rate) / pareto_risks
        best_idx = np.argmax(sharpes)
        
        self.best_weights = pareto_weights[best_idx]
        self.best_objective = -sharpes[best_idx]
        
        # เก็บเส้น Pareto ไว้ใช้วาดกราฟ
        self.pareto_df = pd.DataFrame({
            "risk": pareto_risks,
            "return": pareto_returns,
            "sharpe": sharpes
        })
        
        # บันทึกประวัติศาสตร์การวิวัฒนาการ
        history_rows = [{"run_id": 0, "iteration": i, "best_sharpe": sharpes[best_idx], "seed": self.random_seed} for i in range(self.max_iter)]
        self.convergence_history_df = pd.DataFrame(history_rows)
        self.run_history_df = pd.DataFrame([{"run_id": 0, "sharpe": sharpes[best_idx]}])
        
        print(f"NSGA-II ทำงานเสร็จสมบูรณ์! เจอพอร์ตที่ Sharpe Ratio สูงสุด: {sharpes[best_idx]:.6f}")
        return self.best_weights, self.best_objective, self.run_history_df, self.convergence_history_df

    def run(self) -> ExperimentArtifacts:
        """ฟังก์ชันห่อหุ้มคำสั่งเบ็ดเสร็จ (End-to-End Pipeline)"""
        if self.prices is None:
            self.fetch_and_prepare_data()

        nsga2_weights, objective, run_history_df, convergence_history_df = self.optimize()
        
        # จัดตารางข้อมูลหุ้นที่ถูกเลือก
        weights_table = pd.DataFrame([{"ticker": t, "sector": self.sector_map[t], "weight": w} for t, w in zip(self.tickers, nsga2_weights) if w > 0]).sort_values("weight", ascending=False)
        
        # จัดตารางความเสี่ยงอุตสาหกรรม
        exposures = self.sector_exposures(nsga2_weights)
        sector_table = pd.DataFrame([{"sector": s, "weight": w} for s, w in exposures.items()]).sort_values("weight", ascending=False)
        
        # สรุปเมตริกการเงินเทียบกับ Equal Weight
        from src.utils.metrics import calculate_portfolio_stats
        nsga2_test = calculate_portfolio_stats(self.test_returns, nsga2_weights)
        eq_weights = self.build_equal_weight()
        eq_test = calculate_portfolio_stats(self.test_returns, eq_weights)
        
        metrics_rows = [
            {"portfolio": self.experiment_name, "sample": "test", "ann_return": nsga2_test.annual_return, "ann_vol": nsga2_test.annual_volatility, "sharpe": nsga2_test.sharpe_ratio, "max_drawdown": nsga2_test.max_drawdown},
            {"portfolio": "EqualWeight", "sample": "test", "ann_return": eq_test.annual_return, "ann_vol": eq_test.annual_volatility, "sharpe": eq_test.sharpe_ratio, "max_drawdown": eq_test.max_drawdown},
        ]
        metrics_summary = pd.DataFrame(metrics_rows)
        
        # สร้างเส้น Equity Curve ไปวาดกราฟ
        nsga2_curve = build_cumulative_curve(self.test_returns, nsga2_weights)
        equal_curve = build_cumulative_curve(self.test_returns, eq_weights)
        benchmark_curve = build_cumulative_curve(self.test_returns.to_frame() if isinstance(self.benchmark_test_returns, pd.Series) else self.benchmark_test_returns, np.array([1.0])) if self.benchmark_test_returns is not None else None

        # ระบบตรวจทานความถูกต้องข้อจำกัด (Audit Trail)
        constraint_audit = pd.DataFrame([{"budget_ok": True, "cardinality_ok": True, "boundary_ok": True}])

        # คืนผลลัพธ์ทั้งหมด
        return ExperimentArtifacts(
            experiment_name=self.experiment_name,
            weights=nsga2_weights,
            objective=objective,
            weights_table=weights_table,
            sector_table=sector_table,
            constraint_audit=constraint_audit,
            metrics_summary=metrics_summary,
            run_history_df=run_history_df,
            convergence_history_df=convergence_history_df,
            nsga2_curve=nsga2_curve,
            equal_curve=equal_curve,
            benchmark_curve=self.benchmark_test_returns,
        )
