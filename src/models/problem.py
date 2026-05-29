import numpy as np
from pymoo.core.problem import ElementwiseProblem

class PortfolioProblem(ElementwiseProblem):
    """
    คลาสนี้ทำหน้าที่กำหนด 'เป้าหมาย' (Objectives) และ 'ข้อจำกัด' (Constraints) ของโมเดล
    สืบทอดมาจาก ElementwiseProblem ของไลบรารี Pymoo 
    เพื่อใช้ประเมินความเก่ง (Fitness) ของแต่ละโครโมโซมในประชากร NSGA-II

    [สูตรคณิตศาสตร์: Modern Portfolio Theory]
    เป้าหมายที่ 1 (Min Risk): Minimize f1(w) = sqrt(w^T * Covariance * w)
    เป้าหมายที่ 2 (Max Return): Maximize f2(w) = w^T * Expected_Return
    """
    def __init__(self, optimizer):
        self.opt = optimizer
        n_assets = len(optimizer.tickers)
        
        # โครงสร้าง 2N Encoding: ครึ่งแรกคือน้ำหนัก (Weights), ครึ่งหลังคือคะแนนการเลือก (Selection Scores)
        # ตัวแปร (n_var) = 2 * จำนวนหุ้น
        # เป้าหมาย (n_obj) = 2 (เป้าหมาย 1: เสี่ยงต่ำ, เป้าหมาย 2: กำไรสูง)
        # ข้อจำกัด (n_constr) = 0 (เราใช้เทคนิค Penalty Function ยัดเข้าไปใน Objective แทน)
        super().__init__(n_var=2 * n_assets, n_obj=2, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        ฟังก์ชันนี้จะถูกเรียกซ้ำๆ เป็นแสนครั้งโดย NSGA-II เพื่อให้คะแนนโครโมโซม
        ดังนั้น ต้องเขียนโค้ดแบบ Vectorization (ห้ามใช้ for-loop ในสมการเมทริกซ์) เพื่อให้รันเร็วที่สุด
        """
        n_assets = self.opt.n_assets
        weights = x[:n_assets] # ดึงเอายีนครึ่งแรก (น้ำหนัก) มาประมวลผล
        
        # ----------------------------------------------------
        # 1. การคำนวณเป้าหมาย (Objective Calculations)
        # ----------------------------------------------------
        
        # คำนวณผลตอบแทนคาดหวัง (Expected Return) ด้วย Vector Dot Product
        # สูตร: Return = sum(w_i * mu_i)
        ann_return = float(np.dot(weights, self.opt.mu_train.values))
        
        # คำนวณความเสี่ยง (Portfolio Variance & Volatility) ด้วย Matrix Multiplication
        # สูตร: Variance = w^T * Covariance * w
        variance = float(weights.T @ self.opt.cov_train.values @ weights)
        ann_vol = float(np.sqrt(max(variance, 1e-12))) # ป้องกันค่าติดลบใน Root
        
        # ----------------------------------------------------
        # 2. การคำนวณต้นทุนการทำธุรกรรม (Transaction Costs)
        # ----------------------------------------------------
        # เราจำลองว่าพอร์ตเริ่มต้นคือพอร์ตแบบหารเฉลี่ย (Equal Weight)
        # หาก AI ปรับพอร์ตให้ห่างจาก Equal Weight มากเท่าไหร่ ก็จะเสียค่าธรรมเนียมมากขึ้น
        eq_weights = self.opt.build_equal_weight()
        turnover = float(0.5 * np.sum(np.abs(weights - eq_weights)))
        net_return = ann_return - (turnover * 0.002) # หักค่าคอมมิชชัน 0.20% (20 bps)
        
        # ----------------------------------------------------
        # 3. กลไกบทลงโทษ (Penalty Function สำหรับ Sector Limit)
        # ----------------------------------------------------
        sector_penalty = 0.0
        if self.opt.max_sector_weight is not None:
            # คำนวณว่าแต่ละหมวดอุตสาหกรรมมีน้ำหนักรวมเท่าไหร่
            exposures = self.opt.sector_exposures(weights)
            
            # หากทะลุเพดาน (เช่น 40%) จะเกิดค่าความผิดพลาด (Penalty)
            max_exposure = max(exposures.values())
            sector_penalty = max(0.0, max_exposure - self.opt.max_sector_weight)
        
        # ----------------------------------------------------
        # 4. ส่งผลลัพธ์ออกไปให้ Pymoo (Evaluation Output)
        # ----------------------------------------------------
        # Pymoo คาดหวังการหาค่าต่ำสุด (Minimization) เสมอ
        # ดังนั้น เราจึงเอาค่า net_return ไปติดลบ (-) เพื่อหลอกให้อัลกอริทึมพยายามทำกำไรให้สูงที่สุด
        # ตัวคูณ Penalty ที่ 50.0 มีเป้าหมายเพื่อ "ฆ่า" โครโมโซมที่ทำผิดกฎ Sector Limit ให้สูญพันธุ์ไป
        
        out["F"] = [
            ann_vol + (sector_penalty * 50.0),   # Objective 1: Minimize Risk
            -net_return + (sector_penalty * 50.0) # Objective 2: Maximize Return (via negative)
        ]
