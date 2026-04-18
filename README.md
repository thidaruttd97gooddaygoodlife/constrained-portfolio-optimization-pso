# Constrained Portfolio Optimization using PSO (Mean-Variance)

โปรเจกต์นี้สาธิตการจัดพอร์ตลงทุนภายใต้ข้อจำกัดโลกจริงด้วย **Particle Swarm Optimization (PSO)** บนกรอบ **Mean-Variance**

## Problem Statement
ต้องการหาเวกเตอร์น้ำหนักลงทุน `w` ที่ทำให้ **Sharpe Ratio** สูงสุด ภายใต้เงื่อนไข:

1. **Budget Constraint**: `sum(w) = 1.0`
2. **Boundary Constraint**: `0.00 <= w_i <= 0.25`
3. **Cardinality Constraint**: เลือกหุ้น **exactly 10 ตัว** จากทั้งหมด 30 ตัว (`w_i > 0` แค่ 10 ตัว)

> จุดท้าทาย: เมื่อมี Cardinality ปัญหาจะเป็น combinatorial / non-convex ซึ่งยากสำหรับวิธี deterministic ปกติ


## Outputs 
สคริปต์จะสร้าง:
1. **Efficient Frontier-like Cloud** (random constrained portfolios) + จุดพอร์ต PSO และ Equal Weight
2. **Bar Chart** หุ้น 10 ตัวที่ถูกเลือกและน้ำหนัก
3. **Cumulative Return** เปรียบเทียบ PSO vs Equal Weight (1 ปีล่าสุด)

พร้อมพิมพ์ผลสรุป:
- รายชื่อหุ้นที่เลือกและน้ำหนัก
- ตรวจสอบ constraints ว่าผ่านหรือไม่
- ค่า Return / Volatility / Sharpe ของพอร์ตเทียบ benchmark
- มูลค่าท้ายพอร์ตจาก Backtest
