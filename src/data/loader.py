"""
โมดูลการโหลดและเตรียมข้อมูล (Data Loader & Preprocessing)
------------------------------------------------------
ทำหน้าที่เชื่อมต่อกับ Yahoo Finance (yfinance) เพื่อดึงข้อมูลราคาหุ้นในอดีต
และทำการแปลงข้อมูลดิบ (Raw Prices) ให้กลายเป็น ข้อมูลที่พร้อมเข้าสมการ (Returns)
พร้อมกับแบ่งชุดข้อมูล (Train/Test Split) สำหรับการทดสอบความแม่นยำของโมเดล
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Tuple

def download_adjusted_close(tickers: List[str], benchmark_ticker: str, years: int = 10) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    ฟังก์ชันดึงข้อมูลราคาปิดที่ปรับปรุงแล้ว (Adjusted Close Price)
     Adjusted Close สำคัญมากในการทำ Portfolio เพราะมันรวมการจ่ายปันผล (Dividends) และแตกพาร์ (Stock Splits) แล้ว
    
    พารามิเตอร์:
        tickers: รายชื่อหุ้นที่จะดึง
        benchmark_ticker: ดัชนีอ้างอิง (เช่น ^GSPC คือ S&P 500)
        years: ดึงข้อมูลย้อนหลังกี่ปี
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"กำลังดาวน์โหลดข้อมูลหุ้น {len(tickers)} ตัว จาก Yahoo Finance (ย้อนหลัง {years} ปี)...")
    
    # 1. โหลดข้อมูลหุ้นแบบกลุ่ม (Batch Download) ช่วยให้โหลดเสร็จไวขึ้น
    prices = yf.download(
        tickers=tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True, # ปรับ Adjusted Close อัตโนมัติ
        progress=False,
        group_by="column",
        threads=True,
    )

    if prices.empty:
        raise RuntimeError("เกิดข้อผิดพลาด! ไม่สามารถดาวน์โหลดข้อมูลจาก Yahoo Finance ได้")

    # 2. จัดการโครงสร้างข้อมูล (Data Munging)
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"].copy()
    else:
        close = prices.copy()

    # 3. จัดการข้อมูลสูญหาย (Missing Value Handling / Imputation)
    close = close.dropna(how="all") # ลบวันที่ตลาดปิด (ไม่มีหุ้นตัวไหนเทรดเลย)
    
    # Forward Fill (ffill): ถ้าวันนี้ไม่มีราคา ให้เอาราคาเมื่อวานมาใช้
    # Backward Fill (bfill): ถ้าวันแรกๆ ไม่มีราคา ให้เอาราคาวันถัดไปมาใช้
    close = close.ffill().bfill() 
    
    # ตัดหุ้นที่เพิ่งเข้าตลาดมาไม่นาน (ข้อมูลหายเกิน 20% ของระยะเวลาทั้งหมด) ทิ้งไป
    close = close.dropna(axis=1, thresh=len(close) * 0.8)

    available = [ticker for ticker in tickers if ticker in close.columns]
    prices_df = close[available].copy()

    print(f"กำลังดาวน์โหลดข้อมูลดัชนีอ้างอิง (Benchmark): {benchmark_ticker}...")
    benchmark_raw = yf.download(
        tickers=benchmark_ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    
    # สกัดเฉพาะราคาปิดของ Benchmark
    benchmark_series = benchmark_raw["Close"] if "Close" in benchmark_raw.columns else benchmark_raw
    if isinstance(benchmark_series, pd.DataFrame):
        benchmark_series = benchmark_series.iloc[:, 0]
    benchmark_series = benchmark_series.dropna().rename(benchmark_ticker)

    print(f"ดาวน์โหลดสำเร็จ! หุ้นที่นำไปคำนวณต่อได้: {len(available)}/{len(tickers)} ตัว")
    return prices_df, benchmark_series, available

def prepare_returns(prices: pd.DataFrame, benchmark_prices: pd.Series, split_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    ฟังก์ชันแปลงราคา (Prices) ให้เป็น ผลตอบแทน (Returns) และแบ่งชุดข้อมูล Train/Test
    สมการ: Return_t = (Price_t - Price_{t-1}) / Price_{t-1}
    """
    # .pct_change() คือฟังก์ชันหาอัตราการเปลี่ยนแปลง (ผลตอบแทนรายวัน)
    returns_daily = prices.pct_change().dropna()
    
    # ----------------------------------------------------
    # Train / Test Split (หลักการ Machine Learning)
    # ----------------------------------------------------
    # Train Set: ให้ AI เรียนรู้และหาพอร์ตที่ดีที่สุด (ตั้งแต่แรก จนถึงก่อนวันสุดท้าย 252 วัน)
    train_returns = returns_daily.iloc[:-split_days].copy()
    
    # Test Set: เก็บข้อมูล 252 วันสุดท้าย (ประมาณ 1 ปี) เอาไว้สอบวัดผลว่า AI เทรดจริงแล้วจะรอดไหม
    test_returns = returns_daily.iloc[-split_days:].copy()
    
    benchmark_returns = benchmark_prices.pct_change().dropna()
    # จัด Alignment วันที่ของ Benchmark ให้ตรงกับพอร์ตหุ้น
    benchmark_returns = benchmark_returns.reindex(returns_daily.index).ffill().dropna()
    benchmark_test_returns = benchmark_returns.loc[test_returns.index]
    
    return train_returns, test_returns, benchmark_returns, benchmark_test_returns
