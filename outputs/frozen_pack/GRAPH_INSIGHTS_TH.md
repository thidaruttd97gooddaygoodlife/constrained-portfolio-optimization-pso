# GRAPH INSIGHTS & PRESENTATION NOTES (TH)

## ลำดับการเล่าเรื่องแนะนำ (สไลด์)
1) Data Quality → 2) EDA → 3) Forecasting Validation → 4) PSO Optimization → 5) Constraint Audit → 6) Performance Comparison → 7) Robustness → 8) Limitations

## วิธีอ่านกราฟ + Insight (ไฟล์ในโฟลเดอร์ images)

### 01_normalized_prices_all_assets.png
- **ดูอะไร**: เส้น normalized ของทุกหุ้นจากฐานเดียวกัน (=1)
- **Insight**: การกระจายตัวสูงมาก แปลว่า stock selection และ weighting สำคัญกว่าการถือเท่ากัน
- **พูดบนสไลด์**: "ตลาดมี dispersion สูง จึงมีช่องให้ optimization สร้าง alpha"

### 02_daily_returns_distribution.png
- **ดูอะไร**: รูปทรงการกระจายผลตอบแทนรายวัน
- **Insight**: มีหาง (tail risk) ไม่ควรดูแค่ค่าเฉลี่ย
- **พูดบนสไลด์**: "ความเสี่ยง extreme move มีจริง จึงต้องบริหารความเสี่ยงผ่าน covariance + constraints"

### 03_correlation_heatmap.png
- **ดูอะไร**: คู่สินทรัพย์ที่ correlation สูง/ต่ำ
- **Insight**: diversification เกิดจากการผสมหุ้น correlation ไม่สูงพร้อมกัน
- **พูดบนสไลด์**: "PSO เลือกน้ำหนักบนโครงสร้าง correlation ไม่ใช่ดูรายตัวอย่างเดียว"

### 04_rolling_volatility_top8.png
- **ดูอะไร**: ความผันผวนแบบ rolling 21 วัน
- **Insight**: volatility เปลี่ยนตาม regime จึงต้องทดสอบ out-of-sample
- **พูดบนสไลด์**: "โมเดลถูกประเมินบนช่วงเวลาที่ไม่เท่ากัน เพื่อลด overfit"

### 05_single_asset_price_sma.png
- **ดูอะไร**: ราคาและ SMA feature
- **Insight**: ฟีเจอร์เทคนิคช่วยจับ momentum/reversion

### 06_baseline_forecast_compare.png
- **ดูอะไร**: Actual vs Naive/MA5
- **Insight**: baseline คือ lower bound ที่โมเดลซับซ้อนต้องชนะให้ได้

### 07_forecast_rmse_comparison.png
- **ดูอะไร**: RMSE ของหลายโมเดล
- **Insight**: เลือกโมเดลด้วย metric ไม่ใช่ความซับซ้อน

### 08_residual_over_time.png + 09_residual_distribution.png
- **ดูอะไร**: residual drift และรูปทรง error
- **Insight**: residual กระจุกใกล้ศูนย์ = bias ต่ำลง

### 10_walkforward_actual_vs_predicted.png
- **ดูอะไร**: ผล walk-forward (time-order จริง)
- **Insight**: เป็นหลักฐานสำคัญว่าไม่ leakage

### 11_portfolio_metrics_bar.png
- **ดูอะไร**: ann_return / ann_vol / sharpe ระหว่าง PSO vs Equal
- **Insight**: เปรียบเทียบแบบ risk-adjusted ไม่ใช่ return อย่างเดียว

### 12_efficient_frontier.png
- **ดูอะไร**: cloud พอร์ตสุ่ม + จุด PSO/Equal
- **Insight**: PSO อยู่โซน efficient กว่าในเชิง Sharpe

### 13_selected_weights.png
- **ดูอะไร**: หุ้นที่ถูกเลือกและน้ำหนัก
- **Insight**: strict buy-in 5% ทำให้ cardinality มีความหมายเชิงปฏิบัติ

### 14_cumulative_returns.png
- **ดูอะไร**: เส้นเติบโตเงินลงทุน $10,000
- **Insight**: PSO outperform แบบต่อเนื่องในช่วงทดสอบ

### 15_rf_sensitivity.png
- **ดูอะไร**: Sharpe เมื่อเปลี่ยน risk-free rate
- **Insight**: PSO ยังชนะ benchmark ทุก rf ที่ทดสอบ

### 16_pso_run_history.png
- **ดูอะไร**: Sharpe จากหลาย seed
- **Insight**: solution มีความเสถียร ไม่ได้เป็น one-shot luck

## ประโยคปิดงาน (ใช้พูด 20 วินาที)
"ภายใต้ข้อจำกัดจริง (budget/boundary/cardinality/buy-in) PSO ให้พอร์ตที่ผ่าน constraint ครบและให้ Sharpe สูงกว่า EqualWeight อย่างชัดเจนใน out-of-sample"