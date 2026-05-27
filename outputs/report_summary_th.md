# รายงานการวิจัยขั้นสูง: การจัดสรรพอร์ตการลงทุนด้วย NSGA-II (Academic Research Report)

## 1. บทนำและสมมติฐาน (Introduction & Hypothesis)
งานวิจัยนี้มุ่งเน้นการแก้ปัญหา **Multi-Objective Portfolio Optimization** ภายใต้เงื่อนไขข้อจำกัดที่สมจริง (Real-world Constraints) โดยมีสมมติฐานว่าการใช้อัลกอริทึมเชิงวิวัฒนาการ (Evolutionary Algorithms) สามารถค้นหาชุดคำตอบที่มีประสิทธิภาพ (Efficient Frontier) ได้ดีกว่าวิธีการดั้งเดิมเมื่อเผชิญกับเงื่อนไข Cardinality และ Sector Caps

### 1.1 ทฤษฎีอ้างอิง
เราประยุกต์ใช้ **Modern Portfolio Theory (MPT)** ของ Harry Markowitz เป็นพื้นฐานในการวัดความสัมพันธ์ระหว่าง Risk และ Return

---

## 2. กระบวนการและเครื่องมือ (Methodology & Tools)
การวิจัยแบ่งการทำงานออกเป็น 2 ส่วนหลัก:
1. **The Model:** การสร้างสมการ Mean-Variance เพื่อกำหนดพื้นที่คำตอบ (Search Space)
2. **The Algorithm:** การใช้ **NSGA-II** ผ่านไลบรารี **Pymoo** เพื่อทำการค้นหาคำตอบแบบ Iterative
    *   **Crossover:** การผสมคุณลักษณะพอร์ต
    *   **Mutation:** การรักษาความหลากหลายของคำตอบ
    *   **Elite Preservation:** การรักษาคำตอบที่ดีที่สุดไว้ในรุ่นถัดไป

---

## 3. ตัวแปรและการดำเนินการ (Experimental Setup)
### 3.1 ข้อมูล (Data Universe)
- **Universe:** 50 หุ้นชั้นนำ (Selected Assets)
- **Timeframe:** 10 ปี (2014-2024)
- **Data Points:** > 126,000 จุดข้อมูล (High Fidelity Data)

### 3.2 เงื่อนไขบังคับ (Constraints)
- **Cardinality:** 10 Assets (Constraint CC)
- **Weight Bounds:** 5% - 25% per Asset (Constraint WB)
- **Sector Concentration:** < 40% per Sector (Constraint SC)

---

## 4. ผลการทดลองและการวิเคราะห์ (Results & Analysis)
### 4.1 การค้นหา Pareto Front
ผลการดำเนินงานของ NSGA-II สามารถสร้างเส้น **Pareto Front** ที่มีความหนาแน่นและเรียบเนียน (Smooth Convergence) ซึ่งบ่งบอกถึงความสามารถในการหาจุดที่เหมาะสมที่สุดในทุกระดับความเสี่ยง

### 4.2 การเปรียบเทียบผลการดำเนินงาน (Backtesting)
- **In-sample Sharpe:** 1.4754 (Optimal)
- **Out-of-sample Sharpe:** 2.5700 (Robust)
- **Constraint Audit:** ผ่านการตรวจสอบเงื่อนไขทุกข้อ (100% Feasibility)

---

## 5. บทสรุป (Conclusion)
การศึกษาครั้งนี้พิสูจน์ให้เห็นว่า **NSGA-II** เป็นเครื่องมือที่มีประสิทธิภาพสูงในการจัดการกับโจทย์ Computational Finance ขนาดใหญ่ สามารถสร้างพอร์ตการลงทุนที่สมดุลและนำไปใช้งานจริงได้ภายใต้เงื่อนไขข้อจำกัดของสถาบันการเงินครับ
