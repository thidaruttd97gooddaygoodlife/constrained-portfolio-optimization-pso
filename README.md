# 🚀 Advanced Portfolio Optimization using NSGA-II

This project implements a research-grade Multi-Objective Portfolio Optimization system using the **NSGA-II** evolutionary algorithm. It is designed to find the optimal balance between risk and return (The Pareto Front) while strictly adhering to real-world financial constraints.

## 🏛️ Problem Statement
The goal is to select an optimal portfolio of **10 assets** from a universe of **50 Top S&P 500 stocks** based on 10 years of historical data (~126,000 data points).

### Key Constraints:
1.  **Cardinality:** Exactly 10 stocks selected.
2.  **Buy-in Range:** Weight per stock between 5% and 25%.
3.  **Sector Diversification:** No single industry sector can exceed 40% of the total portfolio weight.
4.  **Budget:** Total weights must sum to exactly 100%.

---

## 🛠️ Methodology & Technology
-   **Model:** Markowitz Mean-Variance Theory (Risk vs. Return).
-   **Algorithm:** **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) via the `pymoo` library.
-   **Data:** 10 years of daily Adjusted Close prices from Yahoo Finance (`yfinance`).
-   **Performance Metrics:** Sharpe Ratio, Annualized Volatility, Max Drawdown, and Sector Exposure.
-   **Validation:** 9-year Training period / 1-year Out-of-sample Testing.

---

## 📂 Project Structure
-   `portfolio_pso.py`: The core optimization engine and reporting suite.
-   `real_stock_datascience_workflow.ipynb`: A step-by-step Data Science notebook for educational use.
-   `outputs/`: Directory containing generated Pareto Front plots, cumulative return charts, and CSV performance summaries.

---


## 📊 Key Results
The model successfully identifies the **Tangency Portfolio** (Max Sharpe) on the **Pareto Front**, significantly outperforming the Equal Weight benchmark in out-of-sample testing while maintaining 100% constraint feasibility.

---
*Created for the Computational Intelligence Final Project (2026).*
