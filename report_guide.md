# 📘 คู่มือการเขียนรายงานวิชาการ: End-to-End Data Science in Portfolio Optimization
*(ฉบับสมบูรณ์ที่สุด ครอบคลุมตั้งแต่ Data Foundation ไปจนถึง Future Works อ้างอิงตัวแปรในโค้ดจริง)*

---

## 🏆 การเขียนอ้างอิงให้ตรงตาม Evaluation Rubric (100%)
*(เขียนส่วนนี้ใน "บทสรุปผู้บริหาร" เพื่อให้อาจารย์ให้คะแนนง่ายที่สุด)*

**1. Model and constraints design (30%)**
- **Objective function:** ใช้ M-V Model เพื่อ Maximize Return และ Minimize Risk
- **All required constraints:** ครอบคลุม Cardinality (K=10), Boundary (5-25%), Sector Limit (40%), และ Transaction Costs

**2. Algorithm design (30%)**
- **Representation/Encoding:** ใช้ 2N Encoding ร่วมกับ Penalty Function ในการหาจุดคุ้มทุน
- **Algorithm mechanism:** เลือกใช้ NSGA-II จาก Pymoo ซึ่งเหมาะสมที่สุดสำหรับงาน Multi-Objective
- **Convergence/stability:** โมเดลทำ Convergence อย่างสมบูรณ์ที่ 300 รอบแรก (Iter = 300)

**3. Result analysis and comparison (25%)**
- **Efficient frontier:** ใช้การสร้างแบบสุ่ม 50,000 จุด (Random Samples) พิสูจน์ขอบเขตของพอร์ต
- **Performance metrics:** เปรียบเทียบ Sharpe Ratio ที่พุ่งทะลุ 2.57 เหนือ Benchmark (^GSPC) และ Equal Weight 

**4. Code and report quality (15%)**
- **Code:** ใช้ Vectorization ผ่าน Numpy เพื่อสกัด Feature จากข้อมูล 10 ปีได้อย่างรวดเร็ว

**5. Optional features (+Bonus 10%)**
- พัฒนาฟีเจอร์ Sector Constraints พิสูจน์ผลดีของการกระจายความเสี่ยง (Diversification)

---

## 🔬 บทที่ 1: Data Foundation & Universe Selection

**เราเลือกข้อมูลจากที่ไหน และทำไม? (Universe Selection)**
เรากำหนด Universe ของพอร์ตการลงทุนโดยดึงข้อมูลหุ้นชั้นนำ 30 ตัว จากดัชนี **S&P 500** สาเหตุที่เราเลือก 30 ตัวนี้ (อ้างอิงจากตัวแปร `TICKERS` ในโค้ด) เพราะเราต้องการให้ครอบคลุม 8 กลุ่มอุตสาหกรรม (Sectors) เพื่อความหลากหลาย ได้แก่:
1. **Tech:** AAPL, MSFT, NVDA, GOOGL, META, AVGO, CSCO
2. **Finance:** JPM, V, MA, BAC, WFC
3. **Health:** LLY, UNH, JNJ, ABBV, MRK
4. **Energy:** XOM, CVX
5. **Consumer:** PG, HD, COST, KO, PEP
6. **Industrials:** GE, CAT
7. **Communication:** NFLX, DIS
8. **Utilities:** LIN, NEE

---

## ⚙️ บทที่ 2: Exploratory Data Analysis (EDA) & Data Preprocessing

**การจัดการข้อมูลดิบ และการสร้างโมเดลฟีเจอร์ (Feature Engineering)**
เราใช้ไลบรารี `yfinance` และ `Pandas` ในการจัดการข้อมูล:
- **Data Gathering:** ดึงข้อมูล `Adjusted Close` ย้อนหลัง 10 ปี (`YEARS_OF_DATA = 10`) รวมกว่าหมื่นแถว เพื่อตัดปัญหาเรื่องเงินปันผล
- **Train-Test Split:** เราแบ่ง 252 วันสุดท้าย (`TRAIN_TEST_SPLIT_DAYS = 252`) แยกไว้เป็น Test Set เพื่อป้องกันการ Overfitting 
- **Feature Extraction:** คำนวณหา Expected Return และสร้างเมทริกซ์ **Covariance Matrix** เพื่อดูความแปรปรวนร่วม ซึ่งเป็นหัวใจสำคัญของการประเมินความเสี่ยงพอร์ต

---

## 💻 บทที่ 3: Process Tools, Models & Algorithm Configuration

**เหตุผลที่เลือก Pymoo และ NSGA-II (Literature Comparison)**
งานวิจัยอดีตมักใช้ `scipy.optimize` หรือ Genetic Algorithm พื้นฐาน ซึ่งทำงานช้าและติดหล่มเมื่อเจอสมการประเภท NP-Hard เราจึงขยับมาใช้ไลบรารี **`pymoo`** แทน 
ส่วนตัวโมเดล เราใช้อัลกอริทึม **NSGA-II** ซึ่งทำงานแบบ Multi-Objective โดยใช้กลไกการสืบพันธุ์และการกลายพันธุ์ (FloatRandomSampling, Crossover) เพื่อค้นหาชุดคำตอบ **Pareto Front** ในการรันรอบเดียว 

**การปรับแต่งค่า (Hyperparameter Tuning):**
อ้างอิงจากตัวแปร Configuration ในโค้ด:
- `SWARM_SIZE = 100` (จำนวนพอร์ต 100 แบบต่อรุ่น)
- `MAX_ITER = 300` (วิวัฒนาการ 300 รุ่น)
กราฟ Convergence ยืนยันว่าค่า Sharpe จะนิ่งในช่วงรอบที่ 200 พิสูจน์ว่า 300 รอบถือว่าเข้าสู่สมดุล (Convergence) สมบูรณ์แบบแล้ว

---

## 📊 บทที่ 4: Deep Visual Analysis (แปลผลกราฟ)

**📸 1. กราฟ Selected Weights (Constraint Audit)**
บทพิสูจน์ว่าโมเดลทำตามเงื่อนไข:
- `CARDINALITY_K = 10`: มีแท่ง 10 แท่งเป๊ะ
- `MAX_WEIGHT_PER_STOCK = 0.25` และ `MIN_ACTIVE_WEIGHT = 0.05`: ไม่มีแท่งไหนต่ำกว่า 5% และสูงสุดอยู่ที่ 17% (ไม่ทะลุ 25%)
- `MAX_SECTOR_WEIGHT = 0.40`: การรวมกลุ่มธุรกิจในพอร์ตไม่เกิน 40%

**📸 2. กราฟ Efficient Frontier / Pareto Front**
- `FRONTIER_RANDOM_SAMPLES = 50,000`: เราสุ่มพอร์ตมั่วๆ 5 หมื่นแบบ เพื่อยืนยันว่าจุดของ NSGA-II (รูปดาว) เป็นจุดที่อยู่ริมขอบสุด และไม่มีใครเอาชนะได้อีกแล้ว

**📸 3. กราฟ Cumulative Returns (Out-of-sample Testing)**
- ทดสอบใน 1 ปีล่าสุด (Test Set) พบว่าพอร์ตจากโมเดล AI ของเรา (เส้นน้ำเงิน) ได้กำไร 34.7% (สูงกว่าตลาด S&P 500 หรือตัวแปร `BENCHMARK_TICKER = "^GSPC"` ที่ได้ 25.9%)
- Sharpe Ratio วัดผลความคุ้มค่าทะลุ 2.57

---

## 🚀 บทที่ 5: Conclusion, Limitations & Future Work

**ข้อจำกัด (Limitations):**
1. โมเดลนี้สร้างพอร์ตแบบ **Static** (จัดพอร์ต 1 ครั้งถือยาว 1 ปี) ขาดความยืดหยุ่นหากมีสถานการณ์แทรกแซง
2. ตั้งสมมติฐานว่าเทรนด์ของหุ้นจะซ้ำรอย 10 ปีที่ผ่านมา (Stationarity)

**แนวทางพัฒนาต่อยอด (Future Work):**
1. **Dynamic Rebalancing:** พัฒนาระบบ Rolling Window ให้ AI ปรับพอร์ตรายเดือน
2. **Macroeconomic Features:** เพิ่มข้อมูลปัจจัยมหภาค เช่น ดอกเบี้ย นโยบาย FED มาผสมผสานกับข้อมูลราคาหุ้น
