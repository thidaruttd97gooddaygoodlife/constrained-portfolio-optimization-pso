"""
โมดูลคำนวณตัวชี้วัดทางการเงิน (Financial Metrics Utility)
------------------------------------------------------
คลาสนี้ทำหน้าที่ประเมินประสิทธิภาพของพอร์ตการลงทุนในโลกความเป็นจริง (Out-of-sample Evaluation)
โดยรวบรวมสูตรคณิตศาสตร์ทางการเงินที่สำคัญ เช่น Sharpe Ratio, Max Drawdown และ Volatility
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class PortfolioStats:
    """โครงสร้างเก็บผลลัพธ์ตัวชี้วัดทางการเงิน (Data Class)"""
    annual_return: float        # ผลตอบแทนคาดหวังรายปี (%)
    annual_volatility: float    # ความเสี่ยง หรือ ความผันผวนรายปี (%)
    sharpe_ratio: float         # อัตราส่วนผลตอบแทนต่อความเสี่ยง (ยิ่งสูงยิ่งดี)
    mean_absolute_deviation: float # ค่าเบี่ยงเบนเฉลี่ยสัมบูรณ์ (MAD - วัดการแกว่งตัว)
    max_drawdown: float         # อัตราการขาดทุนสูงสุดจากจุดพีค (%)
    turnover_rate: float        # อัตราการหมุนเวียนพอร์ต (สะท้อนค่าคอมมิชชัน)
    final_value: float          # มูลค่าเงินในพอร์ตวันสุดท้าย

def calculate_portfolio_stats(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    benchmark_weights: np.ndarray | None = None,
    initial_capital: float = 10000.0,
) -> PortfolioStats:
    """
    ฟังก์ชันคำนวณสถิติทางการเงินของพอร์ต จากผลตอบแทนรายวันและน้ำหนักที่ AI เลือก
    
    [หลักการทำงานระดับ Vectorization]
    การใช้ returns_df.values @ weights (Matrix Multiplication) ทำให้หาผลตอบแทนรายวันของพอร์ต
    ได้ในคำสั่งเดียว โดยไม่ต้องใช้ For-loop ลดเวลาประมวลผลไปได้มหาศาล
    """
    # 1. คำนวณผลตอบแทนรายวันของพอร์ต (Daily Portfolio Returns)
    portfolio_daily = pd.Series(returns_df.values @ weights, index=returns_df.index)
    
    # 2. แปลงเป็นผลตอบแทนรายปี (Annualized Return)
    # สมมติฐาน: ผลตอบแทนเฉลี่ยต่อวัน * จำนวนวันเทรดใน 1 ปี (252 วัน)
    annual_return = float(portfolio_daily.mean() * trading_days)
    
    # 3. แปลงความผันผวนรายวันให้เป็นรายปี (Annualized Volatility)
    # สมมติฐาน: ความเบี่ยงเบนมาตรฐาน (SD) * รากที่สองของจำนวนวันเทรด
    annual_volatility = float(portfolio_daily.std(ddof=1) * np.sqrt(trading_days))
    annual_volatility = max(annual_volatility, 1e-12) # ป้องกันการหารด้วยศูนย์
    
    # 4. คำนวณ Sharpe Ratio (อัตราส่วนชาร์ป)
    # สูตร: (Return - Risk_Free_Rate) / Volatility
    # ความหมาย: เรารับความเสี่ยง 1 หน่วย จะได้กำไรส่วนเพิ่มกลับมากี่หน่วย
    sharpe_ratio = float((annual_return - risk_free_rate) / annual_volatility)
    
    # 5. ค่า MAD (Mean Absolute Deviation) การแกว่งตัวของราคา
    mean_absolute_deviation = float(np.mean(np.abs(portfolio_daily - portfolio_daily.mean())) * trading_days)

    # 6. จำลองเส้นความมั่งคั่ง (Wealth Curve / Equity Curve)
    # เอาเงินต้น (10,000) มาคูณทบต้น (Cumulative Product) ไปเรื่อยๆ ทุกวัน
    curve = initial_capital * (1.0 + portfolio_daily).cumprod()
    
    # 7. คำนวณ Maximum Drawdown (การขาดทุนสูงสุดที่เคยเกิดขึ้นจริง)
    running_peak = curve.cummax()             # หาจุดสูงสุดตลอดกาลที่เคยทำได้ (All-time High)
    drawdown = (curve / running_peak) - 1.0   # หาสัดส่วนที่ร่วงลงมาจากจุดสูงสุด
    max_drawdown = float(drawdown.min())      # หาจุดที่ติดลบหนักที่สุด

    # 8. คำนวณอัตราหมุนเวียนพอร์ต (Turnover Rate)
    # เพื่อประเมินว่าเราต้องเสียค่าคอมมิชชันซื้อขายเยอะแค่ไหน
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

def calculate_benchmark_stats(
    benchmark_returns: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    initial_capital: float = 10000.0
) -> PortfolioStats:
    """เหมือน calculate_portfolio_stats แต่ใช้สำหรับดัชนีตลาด (S&P 500) ที่มีมิติเดียว"""
    annual_return = float(benchmark_returns.mean() * trading_days)
    annual_volatility = float(benchmark_returns.std(ddof=1) * np.sqrt(trading_days))
    annual_volatility = max(annual_volatility, 1e-12)
    
    sharpe_ratio = float((annual_return - risk_free_rate) / annual_volatility)
    mean_absolute_deviation = float(np.mean(np.abs(benchmark_returns - benchmark_returns.mean())) * trading_days)

    curve = initial_capital * (1.0 + benchmark_returns).cumprod()
    running_peak = curve.cummax()
    drawdown = (curve / running_peak) - 1.0
    max_drawdown = float(drawdown.min())

    return PortfolioStats(
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe_ratio=sharpe_ratio,
        mean_absolute_deviation=mean_absolute_deviation,
        max_drawdown=max_drawdown,
        turnover_rate=0.0,
        final_value=float(curve.iloc[-1]),
    )

def build_cumulative_curve(returns_df: pd.DataFrame, weights: np.ndarray, initial_capital: float = 10000.0) -> pd.Series:
    """ฟังก์ชันสกัดเฉพาะเส้นการเติบโตของเงินทุน (Equity Curve) เพื่อนำไปวาดกราฟ"""
    portfolio_daily = pd.Series(returns_df.values @ weights, index=returns_df.index)
    return initial_capital * (1.0 + portfolio_daily).cumprod()
