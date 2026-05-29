"""
ไฟล์การตั้งค่าระบบ (System Configuration)
---------------------------------------
ไฟล์นี้รวบรวมค่าคงที่ (Constants) และพารามิเตอร์ (Parameters) ทั้งหมดที่ใช้ในโปรเจกต์
การแยกไฟล์ตั้งค่าออกมาต่างหาก (Hardcoding avoidance) ช่วยให้ง่ายต่อการปรับจูนพารามิเตอร์ 
(Hyperparameter Tuning) ในอนาคต โดยไม่ต้องเข้าไปแก้โค้ดลึกๆ ในระบบ
"""

from typing import Dict, List

# ---------------------------------------------------------
# [1] Data Universe (ขอบเขตของข้อมูล)
# ---------------------------------------------------------
# รายชื่อหุ้น (Tickers) จำนวน 30 ตัวที่เลือกมาจากดัชนี S&P 500 ใน 8 หมวดหมู่ธุรกิจ (Sectors)
TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "CSCO", # กลุ่ม Tech
    "JPM", "V", "MA", "BAC", "WFC",                         # กลุ่ม Finance
    "LLY", "UNH", "JNJ", "ABBV", "MRK",                     # กลุ่ม Health
    "XOM", "CVX",                                           # กลุ่ม Energy
    "PG", "HD", "COST", "KO", "PEP",                        # กลุ่ม Consumer
    "GE", "CAT",                                            # กลุ่ม Industrials
    "NFLX", "DIS",                                          # กลุ่ม Communication
    "LIN", "NEE"                                            # กลุ่ม Utilities
]

# Mapping ระหว่างชื่อหุ้นและหมวดหมู่ธุรกิจ (Sector Mapping)
# ใช้สำหรับคำนวณข้อจำกัดการกระจุกตัวของอุตสาหกรรม (Sector Constraint)
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "META": "Tech", "AVGO": "Tech", "CSCO": "Tech",
    "JPM": "Finance", "V": "Finance", "MA": "Finance", "BAC": "Finance", "WFC": "Finance",
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health", "MRK": "Health",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Consumer", "HD": "Consumer", "COST": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "GE": "Industrials", "CAT": "Industrials",
    "NFLX": "Communication", "DIS": "Communication",
    "LIN": "Utilities", "NEE": "Utilities"
}

# สีประจำหมวดหมู่ธุรกิจ (ใช้สำหรับการวาดกราฟและ Data Visualization)
SECTOR_COLORS: Dict[str, str] = {
    "Tech": "#002D62",      # สีน้ำเงินเข้ม
    "Energy": "#FFC000",    # สีเหลืองทอง
    "Finance": "#2ca02c",   # สีเขียว
    "Health": "#d62728",    # สีแดง
    "Consumer": "#9467bd",  # สีม่วง
    "Industrials": "#8c564b", # สีน้ำตาล
    "Communication": "#e377c2", # สีชมพู
    "Utilities": "#7f7f7f",   # สีเทา
}

# ---------------------------------------------------------
# [2] Financial Parameters (พารามิเตอร์ทางการเงิน)
# ---------------------------------------------------------
YEARS_OF_DATA = 10                  # ดึงข้อมูลย้อนหลัง 10 ปี
TRAIN_TEST_SPLIT_DAYS = 252         # แบ่งข้อมูล 252 วันสุดท้าย (ประมาณ 1 ปีการซื้อขาย) เป็น Test Set
TRADING_DAYS = 252                  # จำนวนวันทำการซื้อขายใน 1 ปี (ใช้เพื่อทำ Annualization)
RISK_FREE_RATE = 0.02               # อัตราผลตอบแทนที่ไม่มีความเสี่ยง (2% ต่อปี) ใช้คำนวณ Sharpe Ratio

# ---------------------------------------------------------
# [3] Portfolio Constraints (เงื่อนไขข้อจำกัดของพอร์ต)
# ---------------------------------------------------------
CARDINALITY_K = 10                  # บังคับเลือกหุ้นเข้าพอร์ตให้ได้ 10 ตัวเป๊ะๆ
MAX_WEIGHT_PER_STOCK = 0.25         # ห้ามถือหุ้นตัวใดตัวหนึ่งเกิน 25% (ป้องกันความเสี่ยงเฉพาะตัว)
MIN_ACTIVE_WEIGHT = 0.05            # หุ้นที่เลือก ต้องถือขั้นต่ำ 5% (เพื่อให้คุ้มค่าคอมมิชชัน)
MAX_SECTOR_WEIGHT = 0.40            # ห้ามถือหุ้นหมวดหมู่เดียวกันรวมกันเกิน 40% (Sector Concentration Limit)
ACTIVE_THRESHOLD = 1e-6             # ขีดจำกัดทางคณิตศาสตร์ในการปัดเศษเลข 0

# ---------------------------------------------------------
# [4] NSGA-II Optimization Parameters (พารามิเตอร์สมองกล AI)
# ---------------------------------------------------------
SWARM_SIZE = 100                    # จำนวนประชากร (Population Size) ในแต่ละรุ่น
MAX_ITER = 300                      # จำนวนรุ่น (Generations) ที่ให้วิวัฒนาการ
RANDOM_SEED = 42                    # ตั้งค่า Seed เพื่อให้การรันทุุกครั้งได้ผลลัพธ์เดิม (Reproducibility)
BENCHMARK_TICKER = "^GSPC"          # ดัชนีอ้างอิง S&P 500
FRONTIER_RANDOM_SAMPLES = 50_000    # สุ่มพอร์ต 50,000 รูปแบบเพื่อวาดพื้นหลัง Efficient Frontier
