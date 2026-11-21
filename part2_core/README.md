# Part 2 — Core column + Screw + Web app

แบบจำลองส่วนดักน้ำของ DSQ: แกนกลาง moving bed สวนทาง (ก๊าซขึ้น ซีโอไลต์ 3A ลงตามเกลียว)
แบบ steady state พร้อมหน้าเว็บ Streamlit (โหมด check / design)

## ไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `core_parameters.csv` | พารามิเตอร์ทั้งหมด + หน่วย + แหล่งอ้างอิง/สมมติฐาน (อ่านเข้า dict) |
| `gas_properties.csv`, `gas_props.py` | cp (JANAF), ความหนืด (Sutherland + Wilke) ของก๊าซ |
| `zeolite_stub.py` | ตัวแทน Part 1 ตาม `claude/interface-contract.md`: `Zeolite3A` stub (0.1 kg/kg, 6e-3 1/s, −60100 J/mol), `Zeolite3ASipsPreview` (Sips ชั่วคราว), `inlet_gas()` stub, `OutOfRangeError` |
| `core_column.py` | แกน moving bed: for-loop RK4 ทีละช่วงความสูง + while/bisection หา q ขาออก, LDF, สมดุลพลังงาน, Ergun, เช็กเม็ดลอย |
| `screw.py` | rpm = Q_bulk / (π/4·(D²−d²)·pitch·η) |
| `design.py` | โหมด check, design (กวาด D×H → เล็กสุดที่ผ่าน → bisection หา H), sweep อุณหภูมิ/อัตราซีโอไลต์, Excel |
| `plots.py` | กราฟ matplotlib (profiles, criteria bar, heatmap imshow, ΔP/safety factor vs D, line) |
| `app.py` | หน้าเว็บ Streamlit |
| `run_app.bat` | ดับเบิลคลิกเปิดหน้าเว็บบน Windows |
| `base_case_check.py` | รัน base case แบบ check ทั้ง stub และ Part 1 จริง แล้วพิมพ์ตารางเทียบ (+ `base_case_check.csv`) |
| `property_tables.py` | ตาราง lookup ของ q* และ k_LDF (สร้างครั้งเดียว อ่านด้วย bilinear interpolation) — เร็วขึ้นเกือบ 10 เท่ากับ Part 1 จริง, คลาดเคลื่อน < 1e-4 |
| `test_core.py` | 49 tests (assert) รวม integration กับ Part 1 จริง 8 ข้อ — ไม่มีข้อที่ skip ถ้าไม่มี Part 1 เทสต์จะ FAIL พร้อมบอกที่วางไฟล์ — รันด้วย `python test_core.py` หรือ `pytest` |

การเชื่อมกับ Part 1 ยึดตาม project doc `claude/interface-contract.md` (21 พ.ย. 2026):
Part 2 อ่านค่า ΔH จาก `heat_of_adsorption()`, ρ_bed/ε จาก attribute ของ `Zeolite3A`,
ส่งความดันย่อยของน้ำเข้า `loading()/k_ldf()`, เลือก `model=` (sips/gsta/gsta_cap) ได้บนหน้าเว็บเมื่อใช้ Part 1 จริง
และอ่านก๊าซขาเข้าจาก `inlet_gas(...)["F_mol_s"]`

## โครงสร้างโค้ด (ตามเนื้อหา CPE 100)
- ข้อมูลเข้า/ออกเป็น **dictionary** ทั้งหมด: `case` (ค่าที่ผู้ใช้ปรับ), `criteria`, `geom/op/bed` และผลลัพธ์ `res`
  ใช้ `dz.updated_case(case, {"H_m": 1.0})` เพื่อสร้างสำเนาที่เปลี่ยนบางค่า (ของเดิมไม่เปลี่ยน)
- ฟังก์ชันไอโซเทอมส่งเป็น argument (`zeolite.loading`, `zeolite.k_ldf`) แบบ "function as an argument"
- **for-loop** เดินทีละช่วงความสูง (`march_up`), **while + bisection** หา q ขาออก (`shoot`), `match-case` เลือกกรณีความร้อน/นโยบายนอกช่วงข้อมูล
- custom exception: `ThermalPinchError`, `BracketError`, `ShootingError` (`class ...(Exception)` แบบในชีท Files & OOP)
- `try/except` จัดการ `OutOfRangeError` และการหา Part 1 ไม่เจอ
- ช่วงข้อมูลที่ Part 1 ยอมรับ (แยกตามโมเดลไอโซเทอมและ k_LDF) หาเองด้วย **bisection** (`design.zeolite_limits`) ไม่ต้องพิมพ์ตัวเลขเอง
- สร้าง `Zeolite3A()` ของ Part 1 ในโฟลเดอร์ `part1_zeolite/` (Part 1 เปิด `parameters.csv` จากโฟลเดอร์ปัจจุบัน)
- NumPy: array, `table[:, 1]`, boolean mask; pandas: `read_csv`, `DataFrame`, `pivot`, `idxmin`, `to_excel`; matplotlib: `subplots`, `plot`, `bar`, `imshow`, `colorbar`, `axvline`
- เทสต์: `assert` แบบ unit / integration / regression ตามชีท Debugging

**ส่วนที่อยู่นอกเนื้อหาและเหตุผล** (ควรอธิบายได้ตอนนำเสนอ)
| ส่วน | ใช้ที่ | ทำไม |
|---|---|---|
| Streamlit + `@st.cache_data` | `app.py` | อาจารย์ให้ทำหน้าเว็บ; cache เพื่อไม่คำนวณซ้ำเมื่อค่าเดิม |
| `pathlib.Path(__file__)`, `os.path`, `sys.path.append` | `design.py`, `gas_props.py` | หาไฟล์ CSV และโฟลเดอร์ Part 1 จากตำแหน่งไฟล์ ไม่ขึ้นกับโฟลเดอร์ที่รัน และไม่ใช้ `os.chdir` (แชทหลักตัดสิน 21 พ.ย. 2026) |
| Runge–Kutta 4 + substep | `core_column.py` | Euler ต้องใช้ step เล็กมากถึงจะแม่นพอ (เทสต์ลด step ครึ่งหนึ่ง < 1%) |
| แกน log (`set_xscale`, `set_yscale`) | `plots.py` | pH2O ลดจาก ~3 bar เหลือ ~0.03 bar; ΔP เปลี่ยนหลายสิบเท่าตาม D (แท็บในโหมด design) |
| `pd.ExcelWriter` | `design.py` | หลาย sheet ในไฟล์ Excel เดียว |
| `math.isfinite` | `core_column.py` | ตรวจ NaN ระหว่าง march |
| scipy `solve_bvp` | `test_core.py` เท่านั้น | cross-check คำตอบด้วยวิธีอื่น (spec อนุญาต) |

## โหมด design (ตัดสิน 21 พ.ย. 2026)
รันใน terminal: `python design.py` (ราว 1–3 นาที มีบรรทัดบอกความคืบหน้า) → พิมพ์ผลและบันทึก `design_results.xlsx`
1. หาอัตราซีโอไลต์ต่ำสุดที่ T สูงสุด ≤ 250 °C ด้วย bisection (`min_rate_for_temperature`) ที่ขนาดอ้างอิง D 0.30 × H 2.0 m ซึ่งดักน้ำได้หมด จึงเป็นกรณีความร้อนมากที่สุด; มี assert ว่า T ลดลงเมื่ออัตราเพิ่ม
2. ที่อัตรานั้น กวาด D 0.10–0.50 × H 0.25–3.00 m เลือกปริมาตรเล็กสุดที่ผ่าน 5 เกณฑ์ แล้ว bisection หา H (`design_with_rate`)
3. รายงานอัตรานั้นคู่กับขั้นต่ำด้านความจุ (สมดุลกับก๊าซขาเข้า ~161 kg/h) และอัตราส่วน
4. อัตราต่ำสุดเทียบอุณหภูมิซีโอไลต์ป้อน 200–250 °C (`min_rate_vs_feed_temperature`); ไม่มีอัตราที่ผ่าน → รายงานใน note
5. Sensitivity Cp × 0.8 / 1.0 / 1.2 (`cp_sensitivity`, ไม่แสดงบนเว็บ)

เกณฑ์: ดักน้ำ ≥ 95%, แรงก๊าซ ≤ 70% ของน้ำหนักชั้นเม็ด, ΔP ≤ 1 bar, T ≤ 250 °C, **H/D ≥ 2 (ค่าสมมติแบบ rule of thumb ไม่มีแหล่งอ้างอิงเฉพาะ)**
ความหนาแน่น, ε, Cp, ΔH อ่านจาก `zeo` เท่านั้น (ไม่มีค่าสำรองใน `core_parameters.csv`); ดักน้ำเกิน 99.9% แสดงเป็น "> 99.9%"

## หน้าเว็บ (สั้น ๆ)
```powershell
cd "C:\...\repo"
streamlit run part2_core/app.py        # หรือ python -m streamlit run part2_core/app.py
```
- โหมด Check: ใส่ D, H, อัตราซีโอไลต์ → ตอบในไม่กี่วินาที (Zeolite3A + ตาราง q*/k_LDF อยู่ใน `st.cache_resource`)
- โหมด Design: กดปุ่ม → 1–3 นาที (ผลอยู่ใน `st.cache_data` กดซ้ำด้วยค่าเดิมได้ทันที) แสดงขนาดเล็กสุดตามเกณฑ์ + ขนาดแนะนำ (k_LDF × 0.5) + อัตราต่ำสุดเทียบอุณหภูมิป้อน 7 ค่า: 200, 210, 220, 230, 240, 245, 250 °C (ตารางทีละ 5 °C ใช้ `python design.py`)
- ป้ายเตือนเมื่อ T > 250 °C (extrapolate Sips) และเมื่อใช้ Henry's law; residence time เป็น st.info; ดักน้ำเกิน 99.9% แสดง "> 99.9%"
- Gas drag / bed weight = ค่าสูงสุดเฉพาะจุด (ที่ก้นแกน) ของ Ergun dP/dz ÷ น้ำหนักชั้นเม็ดต่อเมตร; ค่าเฉลี่ยทั้งแกนแสดงคู่กัน
- ไฟล์ข้อความทุกไฟล์เปิดด้วย encoding="utf-8" (กัน cp874 บน Windows ภาษาไทย)

## วิธีรัน
```
repo/
  part1_zeolite/            <- Part 1 ตัวจริง (ห้ามแก้): zeolite.py, feed_gas.py, parameters.csv, test_zeolite.py
  part2_core/               <- โฟลเดอร์นี้
```
```bash
cd part2_core
pip install -r requirements.txt
streamlit run app.py
pip install pytest scipy        # สำหรับเทสต์ (scipy ใช้ cross-check เท่านั้น)
pytest                          # หรือ python test_core.py
```
- ต้องใช้ Python ≥ 3.10 (match-case) ถ้า `pip install streamlit` บน Python 3.14 มีปัญหา ให้สร้าง venv ด้วย 3.12/3.13

### Windows (PowerShell ใน VS Code)
```powershell
cd C:\path\to\repo\part2_core           # โฟลเดอร์ที่มี app.py
Set-ExecutionPolicy -Scope Process Bypass   # ถ้า activate venv ไม่ได้ (ปิดหน้าต่างแล้วหายเอง)
..\.venv\Scripts\Activate.ps1             # ถ้า venv อยู่ที่ repo/.venv (เปลี่ยนตามที่สร้างไว้)
python -m pip install -r requirements.txt  # ครั้งแรกครั้งเดียว
python -m streamlit run app.py             # เบราว์เซอร์เปิด http://localhost:8501 เอง
```
- หยุดเซิร์ฟเวอร์: กด `Ctrl + C` ใน terminal
- หรือดับเบิลคลิก `run_app.bat` (หา venv ที่ `part2_core\.venv` หรือ `repo\.venv` ให้เอง)
- ถ้าพิมพ์ `streamlit` แล้วขึ้นว่าไม่รู้จักคำสั่ง ให้ใช้ `python -m streamlit` แทน (แบบด้านบน)

- Deploy บน Streamlit Community Cloud: main file = `part2_core/app.py` ใช้ `requirements.txt` ในโฟลเดอร์เดียวกัน และเลือกเวอร์ชัน Python ใน Advanced settings

## แบบจำลอง
ต่อหน่วยความสูง (z ขึ้นจากทางเข้าก๊าซ), N = อัตราดูดซับ [mol/(s·m)]:

- N = ρ_bulk · A · k_LDF(T, pH2O) · (q* − q) / M_H2O
- dF_H2O/dz = −N  (ก๊าซ)
- dq/dz = −N · M_H2O / ṁ_zeo  (ซีโอไลต์เคลื่อนลง)
- ΔW · dT/dz = N · 60.1 kJ/mol − U·πD·(T − T_wall),  ΔW = W_gas − W_solid [W/K]

สมมติ local thermal equilibrium (ก๊าซกับเม็ดอุณหภูมิเดียวกัน; ระยะถ่ายเทความร้อนของเม็ด 3 mm แค่ไม่กี่ mm)
และ cp คงที่ที่อุณหภูมิก๊าซเข้า → ΔW คงที่ตลอดแกน จึงแยกได้ 3 กรณี (match-case):

| กรณี | เงื่อนไข | อุณหภูมิชั้นเม็ด |
|---|---|---|
| `gas_carries_heat` | W_gas > W_solid | ยึดกับอุณหภูมิก๊าซเข้าที่ก้น แล้วร้อนขึ้นด้านบน Q/ΔW; อุณหภูมิซีโอไลต์ป้อนไม่ช่วยทำให้เย็น |
| `solid_carries_heat` | W_solid > W_gas | ยึดกับอุณหภูมิซีโอไลต์ป้อนที่ยอด แล้วร้อนขึ้นที่ก้น Q/\|ΔW\| |
| `balanced` | \|ΔW\| < 15% | ความร้อนติดอยู่ในแกน ไม่มีคำตอบ steady → `ThermalPinchError` |

วิธีเชิงตัวเลข: shooting เดินขึ้นจากก้น (RK4 + substep ตาม stiffness) แล้ว bisection หา q ขาออกให้ q ที่ยอด = q_in
ถ้าคอลัมน์ยาวจนการเดินเบี่ยงเพราะ round-off จะเริ่มเดินใหม่จากจุดที่สองคำตอบยังตรงกัน (sequential shooting)
แล้ววน while ภายนอกเพื่ออัปเดตโปรไฟล์ความดัน (Ergun) และความร้อนผ่านผนัง

การตรวจสอบ: ดุลน้ำและพลังงานปิด < 1e-5, ลด step ครึ่งหนึ่งผลเปลี่ยน < 1%, ตรงกับคำตอบวิเคราะห์ (stub),
และตรงกับ `scipy.integrate.solve_bvp` (คนละวิธี เริ่มจาก guess เส้นตรง) ถึงทศนิยม 4 ตำแหน่งทั้งสองกรณี

## ผลสำคัญ (Sips preview ที่จำกัดช่วง 200–250 °C เหมือน Part 1 — รันซ้ำกับ Zeolite3A จริงก่อนใส่รายงาน)
ฐานตาม `claude/water-capture-code-spec.md`: X 30%, S 60%, 30 bar, ก๊าซเข้า 250 °C, ซีโอไลต์ป้อน 200 °C,
q_in 0 (สเกล Sips), ΔH −60.1 kJ/mol, เม็ด 3 mm (น้ำเข้า 25.6 kg/h, ก๊าซ 3.68 mol/s, W_gas ≈ 127 W/K)

1. **อุณหภูมิขึ้นตาม Q/|W_gas − W_solid| ไม่ใช่ Q/(W_gas + W_solid)** (ค่าคิดมือ ~100 °C ใช้แบบหลัง)
   - ซีโอไลต์ 200 kg/h (W_solid < W_gas): ชั้นเม็ดร้อนถึง ~457 °C ดักได้แค่ ~58%
   - ประมาณ 410–560 kg/h: W ใกล้กัน ไม่มีคำตอบ steady (สมดุลพอดีที่ ~477 kg/h)
   - ซีโอไลต์ต้องเป็นตัวพาความร้อนดูดซับ (~24 kW) ออกจากแกน
2. **เกณฑ์ T ในแกน ≤ 250 °C (ช่วงข้อมูล Sips) คุมอัตราซีโอไลต์** (D 0.30 × H 2.0 m):
   | ซีโอไลต์ | อุณหภูมิป้อนสูงสุดที่แกนไม่เกิน 250 °C |
   |---|---|
   | 1,500–2,000 kg/h | ไม่มี (ป้อน 200 °C แล้วแกนยัง 257–283 °C) |
   | ~2,200 kg/h | 200 °C พอดี (ขั้นต่ำ) |
   | 2,500 kg/h | 207 °C |
   | 3,000 kg/h (ค่าเริ่มต้น) | 215 °C |
   | 4,000 kg/h | 225 °C |
   ⇒ **ข้อกำหนดของหน่วย regenerate:** ส่งซีโอไลต์กลับ ≥ 2,200 kg/h ที่ ≤ 200–225 °C ตามตาราง (ปุ่มในแท็บ "Criteria & summary" คำนวณให้)
3. **ขนาดจากโหมด design** (3,000 kg/h, ป้อน 200 °C, เกณฑ์ spec 4 ข้อ + H/D ≥ 2): **D 0.20 m × H 0.75 m**
   → ดักน้ำ 95.5%, T สูงสุด 233 °C, ΔP 20 mbar, lift 0.50, เกลียว 14.7 rpm (ป้อนได้สูงสุด 216 °C)
   - D 0.15 m ไม่ผ่านเพราะเม็ดลอย; ΔP ไม่ใช่ตัวคุม
   - ซีโอไลต์อยู่ในแกนแค่ ~17 s และใช้ความจุน้อย — LDF คิดจลนพลศาสตร์ให้แล้ว
4. **q_in (regenerate ไม่ครบเท่าในเปเปอร์) กระทบมากเมื่อแกนเล็ก:**
   | q_in (kg/kg) | 0 | 0.01 | 0.02 | 0.03 | 0.05 |
   |---|---|---|---|---|---|
   | D 0.30 × H 1.0 m | 100% | 99.6% | 99.2% | 98.8% | 97.2% |
   | D 0.20 × H 0.75 m (design) | 95.5% | 92.8% | 89.5% | 85.6% | 76.1% |
   ⇒ ขนาด design ที่ q_in = 0 ไม่มีที่เผื่อ ควรเลือกแกนใหญ่ขึ้น (เช่น D 0.30 × H 1.0 m) หรือกำหนด q_in เผื่อไว้ในเกณฑ์
   - ที่ q_in = 0 น้ำขาออก (< 0.05 bar) มาจาก Henry extrapolation จึงไม่ควรอ้างค่า ppm ขาออก แต่การผ่าน 95% ยังเชื่อได้

## สมมติฐาน / ค่าที่ต้องตรวจ
- `q_in` = 0 บนสเกล Sips ตาม contract: regenerate แบบ PSA ลดเหลือ 1–3 bar(a) + ไล่ด้วยก๊าซเฉื่อยแห้งที่อุณหภูมิเดิม (สภาวะเดียวกับที่ van Kampen RCE 2021 วัด Sips working isotherm) ค่าบวกบน slider = regenerate ไม่ครบเท่าในเปเปอร์
- ห้ามใช้ GSTA เป็นผลหลักคู่กับ q_in = 0 (GSTA เป็น loading สัมบูรณ์ ต้องใช้ loading สมดุลจริงหลัง regenerate ซึ่งเปเปอร์ไม่ได้ให้) — แอปจะเตือน
- `cp_zeolite` 960 J/(kg·K) จาก datasheet Evonik DYNOCEL 643 (3A) ซึ่งน่าจะวัดที่อุณหภูมิห้อง ที่ 250–330 °C อาจสูงกว่านี้ (ทำให้ผลปลอดภัยขึ้น)
- ρ_bed 640 kg/m³, ε 0.372, ΔH −60.1 kJ/mol อ่านจากโมเดลซีโอไลต์ (contract); datasheet 3A ให้ ΔH สูงสุด 1800 BTU/lb H2O ≈ 75 kJ/mol จึงสอดคล้องว่า −60.1 สมเหตุสมผลกว่า −45.95
- k_LDF มีเฉพาะ micropore; เม็ด 3 mm มีความต้านทาน macropore เพิ่ม (Part 1 ประเมิน ~10% ยังไม่ได้ใส่)
- cp/ความหนืดของ DME และ CH3OH เป็นค่าประมาณ (สัดส่วนน้อย ผลกระทบเล็ก)
- ใช้ **Sips ทั้งแกน** (ตกลง 21 พ.ย.) ไม่สลับ Sips ↔ GSTA กลางแกน; GSTA ใช้เทียบเท่านั้น ถ้าแบบไหนแกนเกิน 250 °C ส่วนนั้นเป็น Sips extrapolate (การเลื่อน T แบบ isosteric ให้ค่าเท่ากับสมการ Sips ตรง ๆ) และหน้าเว็บเตือน
- 40 bar เกินช่วงที่ทดสอบตัวดูดซับ (5–30 bar) → หน้าเว็บเตือนว่า extrapolate
- นอกช่วงข้อมูล (policy `clip`): T ใช้ isosteric shift ตาม Sips (60.1 kJ/mol), pH2O < 0.05 bar ใช้ Henry, k_LDF ใช้ Ea 54.2 kJ/mol — แอปบอกสัดส่วนความสูงที่ extrapolate
- ช่องเกลียวเต็มเม็ด (moving packed bed, plug flow), ไม่หักปริมาตรใบเกลียว, ก๊าซกระจายสม่ำเสมอ
- เกณฑ์ lift ใช้ค่าเฉพาะที่สูงสุด (เข้มกว่าค่าเฉลี่ยทั้งชั้น)
- เพิ่มเองนอก spec: เกณฑ์ H/D ≥ 2 (ตั้งเป็น 0 เพื่อปิดได้) และ `pinch_tol` 15%
- ไม่จำลองเตาและลูป recycle (ตาม spec) ก๊าซเข้าแกนมาจาก stoichiometry ของ X, S

## อ้างอิง
- van Kampen et al., React. Chem. Eng. 2021, 6, 244 (Sips working isotherm)
- van Kampen et al., Adsorption 2021, 27, 577–589 (GSTA, LDF, Table 5)
- van Kampen et al., 2020 (SEDMES cyclic benchmark)
- Bortolamasi & Fottner 2001 (screw volumetric efficiency)
- Evonik DYNOCEL 643 technical data sheet (specific heat of 3A)
- NIST-JANAF Thermochemical Tables (cp); Crane TP-410 (Sutherland constants); Wilke 1950 (mixture viscosity); Ergun 1952
