# Constrained Portfolio Optimization using PSO (Mean-Variance)

โปรเจกต์นี้สาธิตการจัดพอร์ตลงทุนภายใต้ข้อจำกัดโลกจริงด้วย **Particle Swarm Optimization (PSO)** บนกรอบ **Mean-Variance**

## Problem Statement
ต้องการหาเวกเตอร์น้ำหนักลงทุน `w` ที่ทำให้ **Sharpe Ratio** สูงสุด ภายใต้เงื่อนไข:

1. **Budget Constraint**: `sum(w) = 1.0`
2. **Boundary Constraint**: `0.00 <= w_i <= 0.25`
3. **Cardinality Constraint**: เลือกหุ้น **exactly 10 ตัว** จากทั้งหมด 30 ตัว (`w_i > 0` แค่ 10 ตัว)

> จุดท้าทาย: เมื่อมี Cardinality ปัญหาจะเป็น combinatorial / non-convex ซึ่งยากสำหรับวิธี deterministic ปกติ

## Methodology
- โหลดข้อมูลราคาย้อนหลัง 5 ปี (รายวัน) ผ่าน `yfinance`
- คำนวณ Daily Return, Annualized Expected Return, Covariance Matrix
- ใช้ PSO optimize เป้าหมาย: `maximize Sharpe = (R - Rf)/sigma` โดยใช้ `Rf = 0.02`
- ใช้ **penalty function** บังคับเงื่อนไข Budget/Boundary/Cardinality
- ใช้ **buy-in threshold** สำหรับหุ้นที่ถูกเลือก: `w_i >= 0.05`
- ใช้ **multi-restart PSO** (หลาย seed) เพื่อลดความเสี่ยงติด local solution
- เปรียบเทียบกับพอร์ต **Equal Weight (1/30)**
- Backtest 1 ปีล่าสุดด้วยเงินตั้งต้น `$10,000`

## Outputs (สำหรับรายงาน)
สคริปต์จะสร้าง:
1. **Efficient Frontier-like Cloud** (random constrained portfolios) + จุดพอร์ต PSO และ Equal Weight
2. **Bar Chart** หุ้น 10 ตัวที่ถูกเลือกและน้ำหนัก
3. **Cumulative Return** เปรียบเทียบ PSO vs Equal Weight (1 ปีล่าสุด)

พร้อมพิมพ์ผลสรุป:
- รายชื่อหุ้นที่เลือกและน้ำหนัก
- ตรวจสอบ constraints ว่าผ่านหรือไม่
- ค่า Return / Volatility / Sharpe ของพอร์ตเทียบ benchmark
- มูลค่าท้ายพอร์ตจาก Backtest

และส่งออกไฟล์ประกอบรายงานใน `outputs/` เช่น:
- `metrics_summary.csv`
- `pso_weights.csv`
- `constraint_audit.csv`
- `pso_run_history.csv`

## Notes for Presentation
- **Why M-V?** เป็นฐานของ Modern Portfolio Theory และใช้ Covariance ได้ชัดเจน
- **Why PSO?** รับมือ Cardinality Constraint ได้ดีใน search space ขนาดใหญ่
- **Expected insight**: Sharpe ของพอร์ต PSO ควรดีกว่า Equal Weight เมื่อข้อจำกัดสมจริง

## Current Strict Setting
เวอร์ชันปัจจุบันตั้งค่า strict แล้ว:
- `MIN_ACTIVE_WEIGHT = 0.05`

จึงทำให้ cardinality มีความหมายเชิงปฏิบัติจริง (ไม่มีน้ำหนักจิ๋วหลอกการนับจำนวนหุ้น)
