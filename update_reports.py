import re

file_path = r'c:\Users\THITHI\ciproject\constrained-portfolio-optimization-pso\portfolio_pso.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# New Presentation Text
new_presentation_text = r'''    text = f"""# Script พูดหน้าห้อง: Advanced Multi-Objective Optimization (NSGA-II)

## สไลด์ 1: The Hypothesis & Framework
สวัสดีครับ งานวิจัยนี้เรายกระดับจากการจัดพอร์ตแบบเดิม มาเป็นการใช้ **Multi-Objective NSGA-II** 
เราไม่ได้มองหาแค่คำตอบเดียว แต่เราต้องการหา **Pareto Front** ซึ่งเป็นกลุ่มคำตอบที่เหมาะสมที่สุด (Optimal Trade-off) ระหว่างความเสี่ยงและผลตอบแทน

**สมมติฐานหลัก:** การใช้ NSGA-II ร่วมกับข้อจำกัดจริง (Realistic Constraints) จะทำให้เราพบพอร์ตที่มีความสมดุล (Tangency Portfolio) 
ที่มี Sharpe Ratio สูงกว่าการจัดพอร์ตแบบดั้งเดิมอย่างมีนัยสำคัญ แม้จะหักลบค่าธรรมเนียมธุรกรรม (Transaction Costs) แล้วก็ตาม

## สไลด์ 2: Data Volume & Scalability
อาจารย์เน้นเรื่องปริมาณข้อมูล งานนี้เราจึงใช้จักรวาลสินทรัพย์ **50 ตัว (Top S&P 500)** 
และเก็บข้อมูลย้อนหลังถึง **10 ปี** รวมแล้วมากกว่า **126,000 ชุดข้อมูล** (50 หุ้น x 2,520 วัน) 
เพื่อพิสูจน์ว่าอัลกอริทึม **NSGA-II** ของเราสามารถรับมือกับโจทย์ขนาดใหญ่ (Scalability) ได้จริง 
การใช้ Pareto Front ช่วยให้เราเห็นภาพรวมของโอกาสในการลงทุนได้ชัดเจนกว่าวิธี Single-Objective ครับ

## สไลด์ 3: Pareto Front & Tangency Portfolio
ในกราฟนี้คือหัวใจของงานเราครับ เส้นจุดสีม่วงถึงเหลืองคือ **Pareto Front** ที่ NSGA-II ค้นพบ 
เราไม่ได้สุ่มเลือกคำตอบ แต่เราใช้หลักการทางคณิตศาสตร์หาจุด **Max Sharpe Ratio** (จุดดาวสีเหลือง) 
ซึ่งเป็นจุดที่ให้ผลตอบแทนต่อหนึ่งหน่วยความเสี่ยงดีที่สุด หรือที่เราเรียกว่า **Tangency Portfolio** 
จุดนี้ให้ค่า Sharpe ฝั่ง Train สูงถึง {float(best_run['sharpe']):.6f} เหนือกว่ากลุ่มจุดสุ่มอย่างชัดเจน

## สไลด์ 4: Real-world Constraints (TC & TL)
สิ่งที่ทำให้งานนี้ "สมจริง" คือเราใส่ข้อจำกัดที่นักลงทุนต้องเจอจริง:
1. **Cardinality (10/50):** คัดเลือกหุ้นที่ดีที่สุด 10 ตัวจาก 50 ตัว
2. **Transaction Lots (TL):** น้ำหนักหุ้นต้องเป็นทศนิยม 2 ตำแหน่ง (1%) เพื่อให้ง่ายต่อการส่งคำสั่งซื้อขายจริง
3. **Sector Constraints:** คุมน้ำหนักแต่ละกลุ่มธุรกิจไม่ให้เกิน 40% เพื่อป้องกัน Systemic Risk

ผลลัพธ์ใน Out-of-sample พบว่าพอร์ต {constrained_artifacts.experiment_name} ยังคงรักษา Sharpe ได้ที่ {float(constrained_test['sharpe']):.4f} 
สามารถทำกำไรได้สม่ำเสมอภายใต้เงื่อนไขโลกจริงครับ

## ปิดท้าย
สรุปคือ งานนี้พิสูจน์ให้เห็นว่า NSGA-II ไม่ได้แค่หาพอร์ตที่ "กำไรเยอะ" แต่หาพอร์ตที่ "เหมาะสมที่สุด" 
ภายใต้เงื่อนไขโลกจริงและข้อมูลขนาดใหญ่ครับ ขอบคุณครับ
"""'''

# New Report Text
new_report_text = r'''    report = f"""# รายงานสรุปผลเชิงวิชาการ (Academic Final Report)

## 1) วัตถุประสงค์และการยกระดับโมเดล
โปรเจกต์นี้ได้พัฒนาจากการจัดพอร์ตแบบ PSO พื้นฐาน มาเป็น **Multi-Objective NSGA-II** 
เพื่อแก้ปัญหาการจัดสรรพอร์ตโฟลิโอที่มีความซับซ้อนสูง (NP-Hard) โดยใช้ไลบรารี `pymoo` 

## 2) ความเพียงพอและความมั่นคงของข้อมูล (Data Integrity)
- **ปริมาณข้อมูล:** ใช้หุ้นรวม **50 ตัว** ย้อนหลัง **10 ปี** 
- **จำนวน Data Points:** ประมาณ **126,000 ชุดข้อมูล** ซึ่งถือเป็นระดับ Big Data 
- **ความแม่นยำ:** การใช้ข้อมูลจำนวนมากช่วยให้ Covariance Matrix มีความเสถียร (Robust) 

## 3) Realistic Constraints & Advanced Modeling
เราได้เพิ่มเงื่อนไขที่สะท้อนถึงตลาดการเงินจริง 4 ด้านหลัก:
1. **Cardinality Constraint (CC):** เลือกหุ้นที่ดีที่สุดเพียง 10 ตัวจาก 50 ตัว
2. **Transaction Lots (TL):** กำหนดหน่วยการซื้อขายที่ 1% (rounding weights)
3. **Transaction Costs (TC):** หักค่าธรรมเนียมธุรกรรม 0.2% ของมูลค่าการปรับพอร์ต
4. **Sector Constraint (SC):** จำกัดเพดานแต่ละกลุ่มธุรกิจที่ 40% เพื่อการกระจายความเสี่ยง (Diversification)

## 4) ผลการวิเคราะห์ Pareto Front
ผลการรัน NSGA-II แสดงให้เห็นถึงเส้น **Pareto Front** ที่ชัดเจน โดยเราระบุจุด **Tangency Portfolio** 
ที่ให้ค่า Sharpe Ratio สูงที่สุดบนเส้น ซึ่งเป็นจุดแนะนำสำหรับการลงทุนจริง

## 5) การประเมินผล Out-of-sample
- **Optimized Portfolio:** Sharpe = {float(constrained_test['sharpe']):.6f}
- **Equal Weight:** Sharpe = {float(equal_test['sharpe']):.6f}
- **Market Benchmark (^GSPC):** เปรียบเทียบกับดัชนี S&P 500 เพื่อพิสูจน์ Alpha

## 6) บทสรุป
โมเดลนี้แสดงให้เห็นว่าการใช้เทคนิค Computational Intelligence ขั้นสูงอย่าง **NSGA-II** 
สามารถจัดการกับข้อจำกัดที่ซับซ้อนและข้อมูลขนาดใหญ่ได้อย่างมีประสิทธิภาพ 
โดยยังคงให้ผลตอบแทนที่ปรับด้วยความเสี่ยงที่น่าพึงพอใจครับ
"""'''

# Replace presentation script text
content = re.sub(r'    text = f"""# Script พูดหน้าห้อง: Advanced Multi-Objective Optimization \(MOPSO\).*?"""', new_presentation_text, content, flags=re.DOTALL)

# Replace report text
content = re.sub(r'    report = f"""# รายงานสรุปผลเชิงวิชาการ \(Academic Final Report\).*?"""', new_report_text, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Reporting scripts updated successfully.")
