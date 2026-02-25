# FINAL FROZEN RESULT PACK (Thai One-page Summary)

## 1) ที่มาโจทย์และเหตุผลการเลือกโมเดล
งานนี้แก้ปัญหา Constrained Portfolio Optimization ภายใต้กรอบ Mean-Variance โดยใช้ PSO เนื่องจากมีข้อจำกัด Cardinality (เลือกหุ้นจำกัดจำนวน) ทำให้เป็นปัญหา non-convex/combinatorial ที่วิธีเชิงกำหนดแบบดั้งเดิมจัดการยากในเวลาจำกัด

## 2) แหล่งข้อมูลและช่วงเวลา
- แหล่งข้อมูลจริง: Yahoo Finance ผ่าน yfinance
- สินทรัพย์: หุ้นขนาดใหญ่ 30 ตัว (S&P500)
- ช่วงเวลา: 5 ปีย้อนหลัง (รายวัน)
- มาตรการป้องกัน leakage: split ตามเวลา (train = ทั้งหมด ยกเว้น 252 วันสุดท้าย, test = 252 วันสุดท้าย)

## 3) เงื่อนไขที่ใช้ในโจทย์ (Realistic Constraints)
- Budget: sum(w)=1
- Boundary: 0 <= w_i <= 0.25
- Cardinality: เลือกหุ้น exactly 10 ตัว
- Buy-in threshold: หุ้นที่ถูกเลือกต้องมีน้ำหนักอย่างน้อย 5%
- No short sell: น้ำหนักไม่ติดลบ

## 4) วิธีการคำนวณ
- ฟังก์ชันวัตถุประสงค์: maximize Sharpe = (Return - Rf)/Volatility, โดย Rf=0.02
- ตัวแก้ปัญหา: PSO (swarmsize=100, maxiter=500)
- Global search robustness: multi-restart 3 seeds (42, 43, 44)
- ใช้ penalty + projection/repair เพื่อบังคับข้อจำกัด

## 5) ผลลัพธ์จริงจากรอบล่าสุด
- Constraint audit:
  - sum_weights = 1.000000
  - max_weight = 0.186785
  - active_count = 10
  - min_active_weight = 0.050000
  - checks: budget=True, boundary=True, cardinality=True, buyin=True

- Portfolio comparison (Out-of-sample test 252 วัน):
  - PSO: annual return=0.3346, annual vol=0.1569, Sharpe=2.0050, total return test=0.2795
  - EqualWeight: annual return=0.2048, annual vol=0.1547, Sharpe=1.1949, total return test=0.1367
  - ส่วนต่างสำคัญ: Sharpe ดีกว่า 0.8102 และผลตอบแทน test ดีกว่า 14.28%

- Forecasting track (single-asset):
  - Best model by RMSE = baseline_ma5 (RMSE=0.012382)

## 6) ข้อสรุปเชิงวิชาการ
ผลลัพธ์ยืนยันว่า PSO สามารถจัดการข้อจำกัดโลกจริง (โดยเฉพาะ cardinality + buy-in) ได้ถูกต้อง และให้ผลตอบแทนต่อความเสี่ยงดีกว่า benchmark แบบ Equal Weight ในข้อมูลทดสอบจริง

## 7) ข้อจำกัดงานและข้อเสนอแนะ
- ยังไม่รวม transaction cost, slippage, market impact
- อาจมี survivorship bias จากชุดหุ้นปัจจุบัน
- แนะนำเพิ่ม walk-forward สำหรับพอร์ต, ทดสอบหลาย market regimes, และ stress test เพิ่มเติม
