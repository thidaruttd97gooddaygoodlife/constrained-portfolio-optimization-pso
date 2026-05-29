"""
ไฟล์จุดเริ่มต้นการทำงานของโปรแกรม (Main Entry Point)
------------------------------------------------
สคริปต์นี้เอาไว้สำหรับสั่งรันโปรแกรมแบบง่ายๆ โดยซ่อนความซับซ้อนของคณิตศาสตร์และ AI
เอาไว้ในโฟลเดอร์ src/ ทั้งหมด (ตามหลักการซ่อนรายละเอียด หรือ Encapsulation ของ OOP)

วิธีใช้งานผ่าน Terminal:
    python main.py
"""

from src.algorithms.optimizer import PortfolioOptimizer

def main():
    print("=====================================================")
    print("🚀 เริ่มต้นระบบ AI จัดการพอร์ตการลงทุน (Modular Version)")
    print("=====================================================")
    
    # ---------------------------------------------------------
    # การทดลองที่ 1: พอร์ตที่โดนคุมความเสี่ยงรายอุตสาหกรรม (Sector Constrained)
    # ---------------------------------------------------------
    constrained_opt = PortfolioOptimizer(experiment_name="NSGA2_Sector_Constrained")
    print("\n[การทดลองที่ 1] รันพอร์ตลงทุนที่มีข้อจำกัด Sector Limit ไม่เกิน 40%")
    constrained_artifacts = constrained_opt.run()
    
    # ---------------------------------------------------------
    # การทดลองที่ 2: พอร์ตเสรี (Unconstrained / Original Baseline)
    # ---------------------------------------------------------
    print("\n[การทดลองที่ 2] รันพอร์ตลงทุนแบบเสรี ไม่มีข้อจำกัดเรื่อง Sector")
    # เปลี่ยนพารามิเตอร์ max_sector_weight เป็น None
    original_opt = PortfolioOptimizer(experiment_name="NSGA2_Original", max_sector_weight=None)
    
    # เราก๊อปปี้ข้อมูลมาจากตัวแรกเลย จะได้ไม่ต้องเสียเวลาโหลด Data จาก Yahoo Finance ใหม่
    original_opt.copy_market_data_from(constrained_opt)
    original_artifacts = original_opt.run()
    
    print("\n=====================================================")
    print("✅ การทดลองทั้งหมดเสร็จสมบูรณ์ ระบบรันผ่านฉลุย!")
    print("=====================================================")

if __name__ == "__main__":
    main()
