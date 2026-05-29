import numpy as np
from pymoo.core.repair import Repair

class PortfolioRepair(Repair):
    """
    คลาสสำหรับซ่อมแซม (Repair Operator) โครโมโซมที่เกิดจากการกลายพันธุ์หรือผสมพันธุ์
    ให้กลับมาอยู่ในขอบเขตที่เป็นไปได้ทางคณิตศาสตร์ (Feasible Region)
    คลาสนี้ใช้แก้ปัญหา Cardinality Constraint (บังคับให้พอร์ตเลือกหุ้น 10 ตัวเป๊ะๆ)

    [อธิบายหลักการ 2N Encoding]
    ปัญหาการเลือกหุ้น k ตัวจาก N ตัว เป็นปัญหา NP-Hard การใช้ตัวแปร 0/1 (Binary) จะทำให้การ Optimize ช้าและหลงทิศได้ง่าย
    ดังนั้นเราจึงจำลองโครโมโซมความยาว 2N:
    - ครึ่งแรก (N): น้ำหนักที่แท้จริง (w)
    - ครึ่งหลัง (N): คะแนนที่ให้บอทจัดอันดับว่าตัวไหนควรถูกเลือก (Selection Score)
    """
    def __init__(self, optimizer):
        super().__init__()
        self.opt = optimizer

    def _do(self, problem, X, **kwargs):
        """
        ฟังก์ชัน _do จะถูกเรียกโดย Pymoo ก่อนที่จะส่งโครโมโซมไปประเมินความเก่งใน PortfolioProblem
        """
        n_assets = self.opt.n_assets
        
        # วนลูปตรวจสอบทุกโครโมโซมในประชากร (Population)
        for i in range(len(X)):
            # ----------------------------------------------------
            # 1. 2N Encoding Decoder
            # ----------------------------------------------------
            raw_weights = X[i, :n_assets]        # ยีนครึ่งแรก: น้ำหนักดิบ (Raw Intensities)
            selection_genes = X[i, n_assets:]    # ยีนครึ่งหลัง: คะแนนการเลือก (Selection Scores)
            
            # ส่งเข้าไปซ่อมแซมและบังคับกฎข้อจำกัด Cardinality และ Boundary
            # ฟังก์ชัน project_weights_with_cardinality จะจัดอันดับยีนคะแนน และคงไว้แค่ 10 อันดับแรก
            weights = self.opt.project_weights_with_cardinality(raw_weights, selection_genes)
            
            # ----------------------------------------------------
            # 2. Transaction Lots Constraint (จำลองการซื้อขายจริง)
            # ----------------------------------------------------
            # บังคับให้น้ำหนักการลงทุนถูกปัดเศษในระดับ 1% เพื่อให้สามารถไปแปลงเป็นจำนวนหุ้นจริงได้ง่าย
            weights = np.round(weights, 2)
            
            # ----------------------------------------------------
            # 3. Budget Constraint (บังคับเงินรวม = 1.0 เสมอ)
            # ----------------------------------------------------
            # เนื่องจากการปัดเศษทศนิยม อาจทำให้ผลรวมน้ำหนักคลาดเคลื่อนไปจาก 1.0 (เช่น กลายเป็น 1.01)
            # เราจึงทำการหารเฉลี่ยสัดส่วนใหม่ (Normalization) เพื่อปรับกลับให้เป็น 1.0 สมบูรณ์แบบ
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            # อัปเดตกลับเข้าไปในโครโมโซมครึ่งแรก เพื่อส่งต่อไปให้คลาส PortfolioProblem ประเมินคะแนน
            X[i, :n_assets] = weights
            
        return X
