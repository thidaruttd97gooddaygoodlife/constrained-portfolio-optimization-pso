# 🎙️ สคริปต์การนำเสนอวิทยานิพนธ์: End-to-End Data Science in Portfolio Optimization
*(โครงสร้างเรียงตามลำดับ Data Science Pipeline พร้อมอธิบายพารามิเตอร์จริงจากโค้ด)*

---

## 🛝 Slide 1: Data Foundation & Universe Selection
**หัวข้อบนสไลด์:** Defining the Universe (The 30 S&P 500 Candidates)

**บทพูด (Speaker Notes):**
"สวัสดีครับ วันนี้เราจะมานำเสนอโครงงานการจัดพอร์ตการลงทุนแบบ End-to-End Data Science ครับ 
เริ่มต้นที่ **Data Foundation** การคัดเลือกข้อมูลพื้นฐาน เรากำหนด Universe ของการลงทุนโดยเลือกหุ้นชั้นนำ 30 ตัวจากดัชนี S&P 500 (เช่น AAPL, MSFT, JPM, LLY, XOM) 

สาเหตุที่เราเลือก 30 ตัวนี้ ไม่ใช่การสุ่มมั่วๆ ครับ แต่เราตั้งใจคัดเลือกให้ครอบคลุม 8 กลุ่มอุตสาหกรรม (Sectors) ได้แก่ Tech, Finance, Health, Energy, Consumer, Industrials, Communication และ Utilities เพื่อเป็นรากฐานในการทำ **Diversification (การกระจายความเสี่ยง)** ที่ดีตั้งแต่จุดเริ่มต้นครับ"

---

## 🛝 Slide 2: Exploratory Data Analysis & Preprocessing Pipeline
**หัวข้อบนสไลด์:** EDA, Feature Extraction & Mathematical Clarity
> *(โชว์รูป `eda_correlation_heatmap.png` หรือ `eda_risk_return_scatter.png` จากโฟลเดอร์ outputs)*

**บทพูด (Speaker Notes):**
"หลังจากได้รายชื่อหุ้น เราเข้าสู่กระบวนการ **Data Preprocessing และ EDA** โดยใช้ไลบรารี `yfinance` และ `Pandas`:
- **Feature Extraction:** ในขั้นตอนนี้ เราสกัดค่าผลตอบแทนเฉลี่ย (Expected Return) และที่สำคัญที่สุดคือการสร้าง **Covariance Matrix** เพื่อหาความสัมพันธ์เชิงความแปรปรวนร่วม ซึ่งเป็นตัวแปรสำคัญที่ใช้คำนวณ Risk ของพอร์ตครับ"

---

## 🛝 Slide 3: Problem Formulation & Constraints
**หัวข้อบนสไลด์:** The Realistic Constraints (NP-Hard Problem)

**บทพูด (Speaker Notes):**
"ในทางทฤษฎี เราต้องการ Maximize Return และ Minimize Risk (มีค่า Risk-Free Rate อ้างอิงที่ 2% หรือ 0.02)
แต่ในโลกความเป็นจริง เราต้องแปลงกฎหมายของกองทุนมาเป็นตัวแปรในโค้ด (Constraints) ดังนี้ครับ:
1. **Cardinality (K = 10):** บังคับให้โมเดลหั่นหุ้นจาก 30 ตัว ให้เหลือ 10 ตัวเป๊ะๆ
2. **Boundary:** หุ้นที่รอดมาได้ ต้องมีน้ำหนักการลงทุนขั้นต่ำ **5% (MIN_ACTIVE_WEIGHT = 0.05)** และห้ามเทหมดหน้าตักเกิน **25% (MAX_WEIGHT_PER_STOCK = 0.25)**
3. **Sector Limit:** ห้ามน้ำหนักรวมของอุตสาหกรรมเดียวกันเกิน **40% (MAX_SECTOR_WEIGHT = 0.40)**

เมื่อเราจับกฎเหล่านี้มัดรวมกัน สมการแบบ Linear ดั้งเดิมจะพังทลายกลายเป็นปัญหา NP-Hard ทันทีครับ"

---

## 🛝 Slide 4: Library & Algorithm Selection
**หัวข้อบนสไลด์:** Why Pymoo? The NSGA-II Evolution

**บทพูด (Speaker Notes):**
"เมื่อเจอปัญหาที่คณิตศาสตร์ธรรมดาแก้ไม่ได้ เราจึงต้องเลือกเครื่องมือทาง Data Science ที่เหมาะสมครับ
เราเลือกใช้ไลบรารี **`pymoo`** ซึ่งเป็นไลบรารีระดับโลกสำหรับงาน Multi-Objective Optimization โดยเฉพาะ ดีกว่าการใช้ `scipy.optimize` ที่มักจะค้างเมื่อเจอ NP-Hard 

เราใช้อัลกอริทึม **NSGA-II** แทนที่จะใช้ PSO หรือ Genetic Algorithm แบบเก่า เพราะ NSGA-II ถูกสร้างมาให้ลากเส้น **Pareto Front** ได้ครบทุกระดับความเสี่ยงในการรันรอบเดียว โดยเราใช้เทคนิค 2N Encoding ร่วมกับ Penalty Function เพื่อควบคุม Constraints ที่เรากล่าวมาทั้งหมดครับ"

---

## 🛝 Slide 5: Algorithm Tuning & Convergence
**หัวข้อบนสไลด์:** Hyperparameter Tuning & Stability

**บทพูด (Speaker Notes):**
"เพื่อให้โมเดลแม่นยำที่สุด เราได้ทำ Parameter Tuning ดังนี้ครับ:
- **Population (SWARM_SIZE) = 100:** เราสร้างพอร์ตจำลอง 100 แบบในแต่ละรุ่น 
- **Generations (MAX_ITER) = 300:** ให้มันทำการวิวัฒนาการ (Crossover & Mutation) จำนวน 300 รุ่น

*(ชี้ที่กราฟ Convergence)* เราจะเห็นได้ว่าในช่วงแรก ค่า Sharpe Ratio กระโดดขึ้นอย่างรวดเร็ว และค่อยๆ ลู่เข้าสู่ความเสถียร (Convergence) และแบนราบในช่วงรอบที่ 200 เป็นต้นไป เป็นการพิสูจน์ทางสถิติว่า AI ค้นพบ Global Optimum สำเร็จและไม่ได้เกิดจากการเดาสุ่มครับ"

---

## 🛝 Slide 6: Result Analysis - Constraint Audit
**หัวข้อบนสไลด์:** Constraint Audit (Did the AI cheat?)
> *(โชว์รูป `selected_weights.png`)*

**บทพูด (Speaker Notes):**
"เรามา Audit ผลลัพธ์กันครับว่า AI ทำตามข้อจำกัด 100% จริงไหม:
1. มี 10 แท่งเป๊ะ (Cardinality K=10)
2. ไม่มีใครต่ำกว่า 5% และหุ้นตัวตึงๆ ถูกเบรกไว้ที่ 17% ไม่ทะลุ 25% แน่นอน (Boundary)
3. ดูสีของแท่งครับ ระบบกระจายเงินไปกลุ่ม Health (สีแดง), Tech (สีน้ำเงินเข้ม), Consumer (สีม่วง) โดยไม่มีกลุ่มไหนรวมกันเกิน 40% (Sector Limit)
AI ของเราเคลียร์ปัญหา NP-Hard ได้อย่างไร้ที่ติครับ"

---

## 🛝 Slide 7: Result Analysis - The Efficient Frontier
**หัวข้อบนสไลด์:** The Efficient Frontier of Risk vs Return
> *(โชว์รูป `pareto_front.png` / `efficient_frontier.png`)*

**บทพูด (Speaker Notes):**
"นี่คือจุดสมดุลครับ จุดนับหมื่นจุดคือการสุ่มพอร์ตแบบ Random (FRONTIER_RANDOM_SAMPLES = 50,000 จุด) 
พอร์ตที่ได้จาก NSGA-II ของเรา (รูปดาว) ไปเกาะอยู่บนเส้นขอบบนซ้ายสุด (Efficient Frontier) ได้อย่างสวยงาม 
เราเลือกจุด **Tangency Portfolio** เพราะมันให้ค่า Sharpe Ratio ที่สูงที่สุด เป็นตัวแทนพอร์ตที่ดีที่สุดไปทดสอบในโลกจริงครับ"

---

## 🛝 Slide 8: Out-of-Sample Performance
**หัวข้อบนสไลด์:** Validating Robustness (Test Set Analysis)
> *(โชว์รูป `cumulative_returns.png`)*

**บทพูด (Speaker Notes):**
"เราเอาพอร์ต AI มาวิ่งในข้อมูล Test Set 252 วันสุดท้ายที่เราซ่อนไว้ครับ
- เส้นสีเหลืองคือดัชนีตลาด S&P 500 (Benchmark)
- เส้นสีน้ำเงินคือ NSGA-II ของเรา

เส้นสีน้ำเงินทำกำไรเติบโตเหนือตลาดชัดเจน (Return 34.7% ชนะตลาดที่ 25.9%) และมี **Sharpe Ratio พุ่งไปถึง 2.57** (สูงกว่า 1 ถือว่าดีมาก)
ที่สำคัญคือเวลาตลาดพัง พอร์ตเราติดลบต่ำสุด (Max Drawdown) เพียง -7.2% ปลอดภัยกว่าตลาดครับ"

---

## 🛝 Slide 9: Limitations & Future Work
**หัวข้อบนสไลด์:** What's Next? (Limitations & Developments)

**บทพูด (Speaker Notes):**
"และในส่วนของข้อจำกัด (Limitations):
1. **Stationarity Assumption:** โมเดลนี้มองข้อมูลย้อนหลัง 10 ปี และเชื่อว่าอนาคตจะเหมือนอดีต ซึ่งโลกจริงอาจมี Black Swan
2. **Static Portfolio:** โมเดลนี้จัดพอร์ตครั้งเดียวถือยาว 1 ปีเต็ม 

แนวทางการพัฒนาต่อ (Future Work):
1. **Dynamic Rebalancing:** ปรับโค้ดให้ทำ Rolling Window ทุกๆ ไตรมาส 
2. **Macroeconomic Features:** เพิ่มข้อมูลเงินเฟ้อ หรือดอกเบี้ย เข้ามาสอน AI เพิ่มเติม

ขอบพระคุณอาจารย์และเพื่อนๆ ทุกท่านที่รับฟังครับ"
