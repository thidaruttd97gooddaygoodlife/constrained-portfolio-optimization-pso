import re
import os

file_path = r'c:\Users\THITHI\ciproject\constrained-portfolio-optimization-pso\portfolio_pso.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update imports
content = content.replace('from pymoo.algorithms.moo.mopso_cd import MOPSO_CD', 'from pymoo.algorithms.moo.nsga2 import NSGA2')

# 2. Update TICKERS to 50
new_tickers = 'TICKERS: List[str] = [\n    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",\n    "JPM", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV",\n    "MRK", "BAC", "CVX", "CRM", "ADBE", "KO", "PEP", "WMT", "AMD", "NFLX",\n    "DIS", "MCD", "TMO", "CSCO", "WFC", "ACN", "ABT", "QCOM", "ORCL", "LIN",\n    "INTU", "INTC", "TXN", "AMGN", "IBM", "ISRG", "GE", "CAT", "HON", "NEE"\n]'
content = re.sub(r'TICKERS: List\[str\] = \[.*?\]', new_tickers, content, flags=re.DOTALL)

# 3. Update settings
content = content.replace('SWARM_SIZE = 200', 'SWARM_SIZE = 100')
content = content.replace('MAX_ITER = 600', 'MAX_ITER = 300')

# 4. Update penalty
content = content.replace('out["F"] = [ann_vol + sector_penalty * 10, -net_return + sector_penalty * 10]', 
                         '# High penalty weight (50) to ensure sector constraints are prioritized\n        out["F"] = [ann_vol + sector_penalty * 50, -net_return + sector_penalty * 50]')

# 5. Update optimize method
old_optimize_start = '    def optimize(self) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:'
new_optimize_logic = '''    def optimize(self) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
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
'''

# Find the end of the optimize function to replace it entirely
optimize_start_pos = content.find(old_optimize_start)
# Look for the next major section marker after the optimize function
optimize_end_pos = content.find('    # =============================', optimize_start_pos)

if optimize_start_pos != -1 and optimize_end_pos != -1:
    content = content[:optimize_start_pos] + new_optimize_logic + '\n\n' + content[optimize_end_pos:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Update Success')
