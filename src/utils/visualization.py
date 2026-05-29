"""
โมดูลการสร้างภาพกราฟิก (Data Visualization)
---------------------------------------
รวมฟังก์ชันสำหรับวาดกราฟทั้งหมดในโปรเจกต์ ด้วย Matplotlib 
เพื่อนำเสนอ Insight หรือ Storytelling ให้อาจารย์และผู้บริหารเข้าใจผลลัพธ์ได้ง่ายที่สุด
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from src.utils.metrics import PortfolioStats

def plot_convergence_curve(convergence_history_df: pd.DataFrame, experiment_name: str, save_path: str) -> None:
    """
    วาดกราฟการลู่เข้าหาคำตอบที่ดีที่สุด (Convergence Curve) ของอัลกอริทึม
    เพื่อพิสูจน์ให้อาจารย์เห็นว่า AI ได้เกิดการ 'เรียนรู้' และ 'พัฒนา' โมเดลไปตามเจเนอเรชันจริงๆ
    """
    plt.figure(figsize=(11, 6))
    
    # วนลูปวาดเส้นลู่เข้าของแต่ละการทดลอง (แต่ละ Random Seed)
    for run_id, run_df in convergence_history_df.groupby("run_id"):
        # อัลกอริทึมหาจุด "อิ่มตัว" (Plateau) หรือจุดที่ AI เลิกฉลาดขึ้นอย่างมีนัยสำคัญ
        final_sharpe = float(run_df["best_sharpe"].iloc[-1])
        threshold = final_sharpe - 0.005
        plateau_rows = run_df.loc[run_df["best_sharpe"] >= threshold, "iteration"]
        plateau_iteration = int(plateau_rows.iloc[0]) if not plateau_rows.empty else int(run_df["iteration"].iloc[-1])
        plateau_value = float(run_df.loc[run_df["iteration"] == plateau_iteration, "best_sharpe"].iloc[0])
        
        label = f"Run {run_id} (seed={int(run_df['seed'].iloc[0])})"
        plt.plot(run_df["iteration"], run_df["best_sharpe"], linewidth=1.6, alpha=0.85, label=label)
        
        # มาร์กจุดอิ่มตัวด้วยวงกลมสีดำ เพื่อโชว์ว่าความเก่งคงที่แล้วที่ Iteration ไหน
        plt.scatter([plateau_iteration], [plateau_value], s=70, facecolors="none", edgecolors="black", linewidths=1.4)

    plt.title(f"Convergence Curve: {experiment_name} (แสดงการลู่เข้าหาค่าออพติมัม)")
    plt.xlabel("Iteration (เจเนอเรชัน)")
    plt.ylabel("Sharpe Ratio (ค่าความเก่ง)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_selected_weights(weights_df: pd.DataFrame, sector_colors: Dict[str, str], experiment_name: str, save_path: str) -> None:
    """
    วาดกราฟแท่ง (Bar Chart) แสดงน้ำหนักหุ้น 10 ตัวที่ถูกเลือก
    โดยมีการใช้ 'สี' แยกตามหมวดธุรกิจ (Sector Colors) อย่างชัดเจน เพื่อแสดงให้เห็นว่า
    โมเดลเราเคารพกฎ Sector Concentration Constraint (ไม่เกิน 40% ในอุตสาหกรรมเดียว)
    """
    plt.figure(figsize=(11, 5))
    
    # ดึงสีตามหมวดหมู่ ถ้าไม่มีให้ใช้สีดำ
    colors = [sector_colors.get(sector, "#000000") for sector in weights_df["sector"]]
    bars = plt.bar(weights_df["ticker"], weights_df["weight"], color=colors)
    
    plt.title(f"Selected Weights (สัดส่วนการลงทุน): {experiment_name}")
    plt.xlabel("Ticker (ชื่อหุ้น)")
    plt.ylabel("Weight (น้ำหนักในพอร์ต)")
    plt.ylim(0.0, max(0.28, float(weights_df["weight"].max()) + 0.03)) # ตั้งเพดานกราฟเผื่อป้ายเปอร์เซ็นต์
    plt.grid(axis="y", alpha=0.25)
    
    # เขียนตัวเลขเปอร์เซ็นต์ (เช่น 25.00%) กำกับไว้บนยอดกราฟแท่งทุกแท่ง
    for bar, weight in zip(bars, weights_df["weight"]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.004, f"{weight:.2%}", ha="center", fontsize=9)

    # สร้างกล่อง Legend อธิบายว่าสีไหนคือธุรกิจอะไร
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=sector)
        for sector, color in sector_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Sector (หมวดธุรกิจ)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_cumulative_returns(nsga2_curve: pd.Series, eq_curve: pd.Series, benchmark_curve: pd.Series | None, experiment_name: str, benchmark_ticker: str, save_path: str) -> None:
    """
    วาดกราฟจำลองการลงทุนจริง (Out-of-sample Backtest Equity Curve)
    โชว์ให้เห็นว่าถ้าลงทุน 10,000 ดอลลาร์จริงๆ แล้วระยะเวลา 1 ปี เงินจะงอกเงยเป็นเท่าไหร่
    """
    plt.figure(figsize=(11, 5))
    
    # เส้นพระเอก (พอร์ต AI สีน้ำเงินเข้ม)
    plt.plot(nsga2_curve.index, nsga2_curve.values, label=f"AI ({experiment_name})", linewidth=2.4, color="#002D62")
    
    # เส้นลูกเมียน้อย (พอร์ตหารเฉลี่ย สีเทา)
    plt.plot(eq_curve.index, eq_curve.values, label="Equal Weight Baseline", linewidth=2.1, color="#7f7f7f")
    
    # เส้นคู่แข่ง (ตลาดหุ้นอ้างอิง S&P 500 สีทอง)
    if benchmark_curve is not None:
        plt.plot(benchmark_curve.index, benchmark_curve.values, label=f"Market ({benchmark_ticker})", linewidth=2.0, color="#FFC000")
        
    plt.title("Out-of-Sample Backtest: จำลองการลงทุนจริงด้วยเงิน $10,000")
    plt.xlabel("Date (วันที่)")
    plt.ylabel("Portfolio Value (มูลค่าพอร์ต - USD)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_pareto_front(pareto_df: pd.DataFrame, experiment_name: str, save_path: str) -> None:
    """
    วาดกราฟจุด Pareto Front (ผลลัพธ์ที่ดีที่สุดทุกๆ ด้าน)
    แกน X คือ ความเสี่ยง (Risk), แกน Y คือ ผลตอบแทน (Return)
    จุดดาวสีเหลืองคือจุดที่ดีที่สุดในสายตานักลงทุน (Max Sharpe Ratio)
    """
    if pareto_df is None or pareto_df.empty:
        return
    plt.figure(figsize=(11, 6))
    
    # จุดเล็กๆ สีตามความเก่ง (viridis)
    plt.scatter(pareto_df["risk"], pareto_df["return"], c=pareto_df["sharpe"], 
                cmap="viridis", s=30, alpha=0.7, label="Pareto Front (Optimal Solutions)")
    
    # ไฮไลต์จุด Max Sharpe Ratio ให้เห็นเด่นชัดด้วยดาวสีเหลืองใหญ่ๆ
    best_idx = pareto_df["sharpe"].idxmax()
    best_p = pareto_df.loc[best_idx]
    plt.scatter(best_p["risk"], best_p["return"], marker="*", color="#FFC000", s=300, 
                edgecolor="black", label=f"Max Sharpe Portfolio ({best_p['sharpe']:.4f})")
    
    plt.title(f"Pareto Front: เส้นโค้งความสมดุลระหว่างความเสี่ยงและกำไร ({experiment_name})")
    plt.xlabel("Annual Volatility (Risk) - ความผันผวนต่อปี")
    plt.ylabel("Annual Return - กำไรต่อปี")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_efficient_frontier(random_df: pd.DataFrame, nsga2_stats: PortfolioStats, eq_stats: PortfolioStats, experiment_name: str, save_path: str) -> None:
    """
    วาดกราฟ Efficient Frontier แบบครบเซ็ต
    นำจุดของพอร์ตแบบสุ่ม (หมอกควันสีๆ 50,000 จุด) มาประทับด้วยจุดของ AI เพื่อโชว์ว่า 
    AI เราเหนือกว่าการเดาสุ่มอย่างชัดเจน (พอร์ต AI จะไปกองอยู่ขอบบนสุดซ้ายสุด)
    """
    plt.figure(figsize=(11, 6))
    
    # กลุ่มเมฆหมอกพอร์ตสุ่ม
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
    
    # ดาวแดง: ตัวแทนพอร์ต AI อัจฉริยะ (อยู่ขอบสุดเสมอ)
    plt.scatter(nsga2_stats.annual_volatility, nsga2_stats.annual_return, marker="*", color="red", s=280, label=experiment_name)
    
    # กากบาทดำ: ตัวแทนพอร์ตบ้านๆ (หารเฉลี่ย)
    plt.scatter(eq_stats.annual_volatility, eq_stats.annual_return, marker="X", color="black", s=180, label="Equal Weight")
    
    plt.title("Efficient Frontier: เปรียบเทียบ AI กับพอร์ตแบบสุ่มนับหมื่นรูปแบบ")
    plt.xlabel("Annual Volatility (Risk) - ความผันผวนต่อปี")
    plt.ylabel("Annual Return - กำไรต่อปี")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
