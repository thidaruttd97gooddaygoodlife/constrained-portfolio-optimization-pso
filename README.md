# Constrained Portfolio Optimization using Self-Adaptive PSO (Mean-Variance)

โปรเจกต์นี้สาธิตการจัดพอร์ตลงทุนภายใต้ข้อจำกัดโลกจริงด้วย Self-Adaptive Particle Swarm Optimization (PSO) บนกรอบ Mean-Variance โดยยกระดับทั้งความเสถียรของอัลกอริทึม ความสมจริงเชิงการเงิน และคุณภาพการรายงานเชิงวิชาการ

## Problem Statement
ต้องการหาเวกเตอร์น้ำหนักลงทุน `w` ที่ทำให้ Sharpe Ratio สูงสุด ภายใต้เงื่อนไข:

1. Budget Constraint: `sum(w) = 1.0`
2. Boundary Constraint: `0.00 <= w_i <= 0.25`
3. Cardinality Constraint: เลือกหุ้น exactly 10 ตัวจากทั้งหมด 30 ตัว
4. Buy-in Constraint: หุ้นที่ถูกเลือกต้องมีน้ำหนักอย่างน้อย `0.05`
5. Sector Constraint: น้ำหนักรวมต่อ sector ห้ามเกิน `0.40`

จุดท้าทายคือเมื่อมี cardinality และ sector diversification ปัญหาจะกลายเป็น non-convex และ combinatorial ซึ่งยากสำหรับวิธี deterministic ทั่วไป

## Methodology
- โหลดข้อมูล Adjusted Close ย้อนหลัง 5 ปีจาก `yfinance`
- แบ่งข้อมูลตามเวลาเป็น train 4 ปี และ out-of-sample test 1 ปีล่าสุด
- คำนวณ Daily Return, Annualized Expected Return, และ Covariance Matrix ตาม Mean-Variance Theory
- ใช้ 2N encoding เพื่อแยก gene สำหรับน้ำหนักดิบและ gene สำหรับการเลือกหุ้น
- ใช้ Self-Adaptive PSO โดยลด inertia weight จาก `0.9` ลงเป็น `0.4` แบบเชิงเส้นเพื่อลด oscillation และทำให้ช่วงท้ายลู่เข้าดีขึ้น
- ใช้ penalty function บังคับ Budget, Boundary, Cardinality, Buy-in, และ Sector constraints
- ใช้ multi-restart 5 รอบเพื่อเพิ่มความทนทานต่อ local optima
- เปรียบเทียบกับ benchmark แบบ Equal Weight (1/30)
- วิเคราะห์ผลทั้งเชิง Sharpe, MAD, Max Drawdown, Turnover Rate และ out-of-sample backtest

## Code Structure
- `portfolio_pso.py`: โค้ดหลักแบบ class `PortfolioPSO` ครบ Data -> Optimize -> Compare -> Backtest -> Plot -> Export
- `real_stock_datascience_workflow.ipynb`: workflow data science และ frozen pack สำหรับส่งงาน
- `requirements.txt`: dependency ที่ต้องใช้

## Outputs
สคริปต์จะสร้างผลลัพธ์ใน `outputs/` ดังนี้

### CSV
- `metrics_summary.csv`: เปรียบเทียบ PSO vs Equal Weight ทั้ง train/test พร้อม Annual Return, Volatility, Sharpe, MAD, Max Drawdown, Turnover Rate
- `pso_weights.csv`: น้ำหนักหุ้นที่เลือกพร้อม sector
- `constraint_audit.csv`: ตรวจ Budget, Boundary, Cardinality, Buy-in, Sector constraints
- `sector_allocations.csv`: น้ำหนักรวมราย sector เทียบกับ limit
- `pso_run_history.csv`: สรุปผล multi-restart 5 รอบ
- `convergence_history.csv`: best fitness และ best Sharpe ทุก iteration
- `pso_backtest_curve.csv`: เส้นมูลค่าพอร์ต PSO ในช่วง test
- `equal_backtest_curve.csv`: เส้นมูลค่าพอร์ต benchmark ในช่วง test

### PNG
- `convergence_curve.png`: พิสูจน์การลู่เข้าของ Self-Adaptive PSO
- `selected_weights.png`: allocation bar chart แยกสีตาม sector
- `cumulative_returns.png`: out-of-sample cumulative return เทียบ benchmark
- `efficient_frontier.png`: random feasible cloud เทียบกับจุด PSO และ Equal Weight

## Notes for Presentation
- Why Mean-Variance: เป็นฐานของ Modern Portfolio Theory และใช้ covariance จับความสัมพันธ์ระหว่างสินทรัพย์ได้ชัดเจน
- Why Self-Adaptive PSO: inertia ที่ลดลงตามเวลา ทำให้ช่วงแรก search กว้างและช่วงท้ายนิ่งขึ้น ลดปัญหากราฟฟันปลา
- Why Sector Constraints: ลด systemic risk จากการกระจุกตัวในอุตสาหกรรมเดียว
- Why Out-of-Sample Test: ใช้ข้อมูล 1 ปีล่าสุดที่โมเดลไม่เคยเห็นเพื่อพิสูจน์ว่าไม่ได้ overfit
- Why MAD in analysis: ใช้โชว์ความเข้าใจว่าความเสี่ยงไม่จำเป็นต้องอธิบายด้วย variance เพียงอย่างเดียว

## Current Strict Setting
- `MIN_ACTIVE_WEIGHT = 0.05`
- `MAX_WEIGHT_PER_STOCK = 0.25`
- `MAX_SECTOR_WEIGHT = 0.40`
- `PSO_RESTARTS = 5`
- `INERTIA_WEIGHT = 0.9 -> 0.4`

การตั้งค่านี้ทำให้ cardinality และ diversification มีความหมายเชิงปฏิบัติจริง ไม่ใช่แค่ผ่านเงื่อนไขในเชิงตัวเลข
