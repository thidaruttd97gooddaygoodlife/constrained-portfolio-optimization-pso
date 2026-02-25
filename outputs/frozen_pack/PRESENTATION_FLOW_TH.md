# PRESENTATION FLOW (TH) - 5 to 7 นาที

## Slide 1: Problem & Why CI
- โจทย์: Mean-Variance + Cardinality + Buy-in
- เหตุผลใช้ PSO: ปัญหา non-convex/combinatorial

## Slide 2: Data & Method
- แหล่งข้อมูลจริง: Yahoo Finance, 5 ปี, 30 หุ้น
- ป้องกัน leakage: time-based split, out-of-sample test

## Slide 3: EDA Evidence
- ใช้กราฟ 01, 02, 03
- เน้น dispersion + correlation structure

## Slide 4: Modeling & Validation
- ใช้กราฟ 06, 07, 10
- บอกว่า baseline/model/walk-forward ถูกต้องตามหลัก

## Slide 5: Portfolio Optimization Core
- constraints: budget=1, 0<=w<=0.25, exactly 10 assets, min 5%
- multi-restart PSO for robustness

## Slide 6: Results
- ใช้กราฟ 12, 13, 14
- ตัวเลขหลักจาก csv/06_portfolio_metrics.csv

## Slide 7: Robustness & Limitations
- กราฟ 15, 16
- พูด limitation: transaction cost/slippage/survivorship bias

## Slide 8: Conclusion
- PSO outperform benchmark (risk-adjusted)
- constraints ผ่านครบจาก csv/07_constraint_audit.csv