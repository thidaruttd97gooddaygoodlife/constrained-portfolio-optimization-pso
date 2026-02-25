# PRESENTATION SCRIPT (TH) — 5–7 นาที

> ใช้สคริปต์นี้พูดหน้าห้องได้ทันที โดยอ้างอิงกราฟและผลจริงใน `outputs/frozen_pack`

## 0) เปิดเรื่อง (20–30 วินาที)
สวัสดี งานนี้เป็นการแก้ปัญหา **Constrained Portfolio Optimization** โดยใช้ **Computational Intelligence (PSO)** บนกรอบ **Mean-Variance** เพื่อหาพอร์ตที่ให้ผลตอบแทนต่อความเสี่ยงดีที่สุด ภายใต้ข้อจำกัดโลกจริง เช่น จำกัดน้ำหนักต่อหุ้นและจำกัดจำนวนหุ้นที่ถือ

---

## 1) Problem Statement + เหตุผลที่ใช้ CI (40–60 วินาที)
โจทย์ของเราไม่ใช่แค่ maximize return แต่ต้อง balance ระหว่าง return กับ risk ภายใต้ข้อจำกัดจริง:
- Budget: ผลรวมน้ำหนักต้องเท่ากับ 1
- Boundary: น้ำหนักแต่ละหุ้นอยู่ในช่วง 0 ถึง 25%
- Cardinality: เลือกหุ้น exactly 10 ตัว
- Buy-in: หุ้นที่เลือกต้องมีน้ำหนักอย่างน้อย 5%

พอมี Cardinality + Buy-in ปัญหาจะเป็น non-convex/combinatorial จึงเหมาะกับ metaheuristic อย่าง PSO มากกว่าวิธี deterministic ปกติ

---

## 2) ข้อมูลและความถูกต้องเชิง Data Science (50–70 วินาที)
แหล่งข้อมูลเป็นข้อมูลจริงจาก Yahoo Finance ผ่าน yfinance
- หุ้น 30 ตัว (large-cap)
- ย้อนหลัง 5 ปีแบบรายวัน

ความถูกต้องเชิงวิธีการ:
- ตรวจ Data Quality ก่อน (missing/duplicate/time continuity)
- ทำความสะอาดข้อมูลด้วย forward-fill + ตัดแถว missing คงค้าง
- แยก train/test แบบเวลา (time-based split) ป้องกัน leakage

**โชว์กราฟ**
- `images/01_normalized_prices_all_assets.png`
- `images/02_daily_returns_distribution.png`
- `images/03_correlation_heatmap.png`

ประโยคสรุป: “ข้อมูลมี dispersion และ correlation structure ชัดเจน จึงมีเหตุผลที่ optimization จะช่วยได้มากกว่าถือเท่ากัน”

---

## 3) โมเดลและฟังก์ชันวัตถุประสงค์ (45–60 วินาที)
เราใช้ Mean-Variance + Sharpe maximization:
- Return พอร์ต: $R_p = w^T\mu$
- Risk พอร์ต: $\sigma_p = \sqrt{w^T\Sigma w}$
- เป้าหมาย: $\max \text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$ โดยกำหนด $R_f=0.02$

การแก้ปัญหา:
- PSO (swarm 100, iter 500)
- ใช้ penalty + projection/repair เพื่อบังคับ constraints
- รันหลาย seed (multi-restart) เพื่อเพิ่ม robustness ของ global search

---

## 4) หลักฐานว่า constraints ผ่านจริง (30–40 วินาที)
จาก `csv/07_constraint_audit.csv`:
- sum_weights = 1.0
- max_weight <= 0.25
- active_count = 10
- min_active_weight >= 0.05
- budget/boundary/cardinality/buyin = True ทั้งหมด

ประโยคสรุป: “โมเดลไม่ได้แค่ได้ผลตอบแทนดี แต่ผ่านข้อกำหนดโลกจริงครบทุกข้อ”

---

## 5) ผลลัพธ์หลัก: PSO vs EqualWeight (60–80 วินาที)
จาก `csv/06_portfolio_metrics.csv`:
- PSO: annual return สูงกว่า, Sharpe สูงกว่า
- EqualWeight: เป็น baseline แบบ traditional

**โชว์กราฟหลัก**
- `images/12_efficient_frontier.png`
- `images/13_selected_weights.png`
- `images/14_cumulative_returns.png`

วิธีพูด:
1) Efficient Frontier: “จุด PSO อยู่ในบริเวณที่คุ้มความเสี่ยงมากกว่า”
2) Selected Weights: “ถือ 10 ตัวจริง และมีขั้นต่ำ 5% ทุกตัว ทำให้ cardinality มีความหมายเชิงปฏิบัติ”
3) Cumulative Return: “เงิน $10,000 เติบโตได้ดีกว่า EqualWeight ในช่วงทดสอบจริง”

---

## 6) Robustness และความน่าเชื่อถือ (35–50 วินาที)
จาก `csv/05_pso_run_history.csv` และ `images/16_pso_run_history.png`:
- หลาย seed ให้ผล Sharpe ใกล้เคียงกัน

จาก `csv/08_rf_sensitivity.csv` และ `images/15_rf_sensitivity.png`:
- เปลี่ยน risk-free rate แล้ว PSO ยังเหนือกว่า Equal

ประโยคสรุป: “ผลลัพธ์ไม่ใช่โชคจากรอบเดียว แต่ยังคงแนวโน้มเดิมเมื่อเปลี่ยนเงื่อนไขสำคัญ”

---

## 7) Limitations + งานต่อยอด (30–40 วินาที)
ข้อจำกัด:
- ยังไม่รวม transaction cost, slippage, market impact
- อาจมี survivorship bias

งานต่อยอด:
- เพิ่ม transaction-cost-aware optimization
- ทำ walk-forward optimization ฝั่งพอร์ต
- ทดสอบหลาย market regimes

---

## 8) ปิดการนำเสนอ (15–20 วินาที)
สรุป: ภายใต้ข้อจำกัดจริงที่เข้มงวด PSO สามารถหาพอร์ตที่ผ่าน constraints ครบ และให้ผลตอบแทนต่อความเสี่ยงดีกว่า EqualWeight บนข้อมูลจริงแบบ out-of-sample

---

## Q&A ที่อาจโดนถาม (ตอบสั้น)
1) **ทำไมไม่ใช้ QP?**
- เพราะมี cardinality + buy-in ทำให้เป็น combinatorial/non-convex

2) **มั่นใจได้ยังไงว่าไม่ overfit?**
- ใช้ time-based split + out-of-sample test + multi-seed robustness

3) **ทำไมต้องมี buy-in 5%?**
- ป้องกันน้ำหนักจิ๋วหลอกการนับ cardinality และสะท้อนการลงทุนจริง

4) **ทำไมใช้ Sharpe?**
- เป็นมาตรฐานวัดผลตอบแทนต่อหนึ่งหน่วยความเสี่ยงและเหมาะกับการเทียบพอร์ต
