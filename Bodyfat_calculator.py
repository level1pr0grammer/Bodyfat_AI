#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, numpy as np, math, sys, time
import torch, torch.nn as nn
from torchvision import models, transforms

# ======== CONFIG ========
MODEL_PATH = "artifacts/gender_resnet18.pt"   # พาธโมเดลเพศ (.pt)
CAM_INDEX  = 0
STILL_FRAMES = 18           # ต้องนิ่งติดต่อกันกี่เฟรมจึงแคป
MOTION_THRESH = 2.0         # ค่าความต่างเฉลี่ย (0-255) ที่ถือว่านิ่ง (ยิ่งเล็กยิ่งเข้มงวด)
SIDE_SCALE = 0.88           # ประมาณความกว้างด้านข้างจากภาพหน้าตรง (0.85-0.9 แนะนำ)

# ======== MediaPipe ========
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# ======== “ช่วงเปอร์เซ็นต์ไขมันแบบคนทั่วไป” (อิงภาพตัวอย่าง) ========
# ไม่ใช่มาตรฐานทางการแพทย์ ใช้เพื่อการสื่อสาร/เปรียบเทียบ
BFP_BANDS = {
    "male": [
        ("Athlete/Lean (8–12%)",          8.0, 12.9),
        ("Fit/Normal (13–19%)",          13.0, 19.9),  # ถ้าต้องการ 15–19 เป๊ะ → (15.0, 19.0)
        ("Average/High (20–24%)",         20.0, 24.9),
        ("Overweight (25–29%)",           25.0, 29.9),
        ("Obese (≥30%)",                  30.0, 99.9),
    ],
    "female": [
        ("Athlete/Lean (15–19%)",         15.0, 19.9),
        ("Fit/Normal (20–27%)",           20.0, 27.9),
        ("Average/High (28–34%)",         28.0, 34.9),
        ("Overweight (35–39%)",           35.0, 39.9),
        ("Obese (≥40%)",                  40.0, 99.9),
    ]
}

# จุดอ้างอิงไว้บอกว่า “ใกล้ภาพตัวอย่าง ~x%”
BFP_REF_POINTS = {
    "male":   [8, 12, 15, 20, 25, 30, 35],
    "female": [15, 20, 25, 30, 35, 40, 45],
}

# ======== Math / Helpers ========
def ramanujan_circumference(a, b):
    if a <= 0 or b <= 0: return 0.0
    h = ((a-b)**2)/((a+b)**2)
    return math.pi*(a+b)*(1 + (3*h)/(10+math.sqrt(4-3*h)))

def clamp_pct(v): return max(3.0, min(60.0, v))

def rfm_percent(height_cm, waist_circ_cm, sex):
    base = 64 if sex.lower().startswith('m') else 76
    if waist_circ_cm <= 0: return 0.0
    return clamp_pct(base - 20.0*(height_cm/waist_circ_cm))

def navy_percent(height_cm, waist_circ_cm, neck_circ_cm, sex, hip_circ_cm=None):
    h = height_cm/2.54
    w = waist_circ_cm/2.54
    n = neck_circ_cm/2.54 if neck_circ_cm else None
    if sex.lower().startswith('m'):
        if n is None or w-n <= 0: return None
        return clamp_pct(86.010*math.log10(w-n) - 70.041*math.log10(h) + 36.76)
    else:
        if n is None or hip_circ_cm is None or (w+hip_circ_cm/2.54-n) <= 0: return None
        return clamp_pct(163.205*math.log10(w+hip_circ_cm/2.54-n) - 97.684*math.log10(h) - 78.387)

def bmr_katch_mcardle(weight_kg, bf_pct):
    lbm = weight_kg*(1 - bf_pct/100.0)
    return 370.0 + 21.6*lbm

def body_mask(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        m = seg.process(rgb).segmentation_mask
        bw = (m*255).astype(np.uint8)
        _, bw = cv2.threshold(bw, 150, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), 1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), 1)
        return bw

def head_to_feet_pixels(lm, H):
    head_candidates = [lm[i].y*H for i in [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value
    ]]
    head_y = min(head_candidates)
    foot_candidates = [lm[i].y*H for i in [
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_HEEL.value,
        mp_pose.PoseLandmark.RIGHT_HEEL.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    ]]
    foot_y = max(foot_candidates)
    return max(0.0, foot_y - head_y), head_y, foot_y

def levels_from_pose(lm, H, W):
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    sh_y = ((ls.y + rs.y)/2.0)*H
    hp_y = ((lh.y + rh.y)/2.0)*H
    torso = hp_y - sh_y
    neck_y  = np.clip(sh_y - 0.20*torso, 0, H-1)
    waist_y = np.clip(sh_y + 0.60*torso, 0, H-1)
    hip_y   = np.clip(hp_y + 0.10*torso, 0, H-1)
    return float(neck_y), float(waist_y), float(hip_y)

def linescan_width(mask, y_pix:int):
    y = int(round(y_pix))
    if y < 0 or y >= mask.shape[0]: return 0
    xs = np.where(mask[y,:] > 0)[0]
    return int(xs.max()-xs.min()) if xs.size>=2 else 0

# ======== Mapping/Label helpers ========
def bf_band_label(sex: str, bf_pct: float) -> str:
    s = "male" if sex.lower().startswith("m") else "female"
    for name, lo, hi in BFP_BANDS[s]:
        if lo <= bf_pct <= hi:
            return name
    return "—"

def bf_ref_nearest(sex: str, bf_pct: float) -> float:
    s = "male" if sex.lower().startswith("m") else "female"
    refs = BFP_REF_POINTS[s]
    return float(min(refs, key=lambda r: abs(r - bf_pct)))

# ======== Gender model ========
def load_gender_model(path, device):
    ckpt = torch.load(path, map_location=device)
    img_size = ckpt.get("img_size", 224)
    class_to_idx = ckpt.get("class_to_idx", {"female":0,"male":1})
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, tfm, idx_to_class, img_size

@torch.no_grad()
def predict_gender(model, tfm, frame_bgr, device, img_size, idx_to_class):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    x = tfm(rgb).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(prob))
    return idx_to_class[idx], float(prob[idx])

# ======== UI Drawing (English text to avoid ??? in OpenCV) ========
def draw_header_footer(img, header_text, footer_text):
    H, W = img.shape[:2]
    # header
    cv2.rectangle(img, (0,0), (W,40), (0,0,0), -1)
    cv2.putText(img, header_text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,255,80), 2, cv2.LINE_AA)
    # footer
    cv2.rectangle(img, (0,H-36), (W,H), (0,0,0), -1)
    cv2.putText(img, footer_text, (10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def draw_guide_box(img, x0,y0,x1,y1):
    # กรอบเขียว + มุมกลมเล็ก ๆ
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    for (cx,cy) in [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]:
        cv2.circle(img, (cx,cy), 6, (0,255,0), 2)

# ======== Main ========
def main():
    # 1) รับอินพุตจำเป็น
    try:
        weight_kg = float(input("น้ำหนัก (kg): "))
        height_cm = float(input("ส่วนสูง (cm): "))
    except:
        print("กรอกตัวเลขไม่ถูกต้อง"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, idx_to_class, img_size = load_gender_model(MODEL_PATH, device)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"เปิดกล้องไม่ได้ index={CAM_INDEX}"); sys.exit(2)

    print("ยืนให้อยู่ในกรอบแล้ว 'โปรดอยู่นิ่งๆ'… ระบบจะถ่ายอัตโนมัติเมื่อคงที่")

    prev_gray = None
    still_count = 0
    captured = None

    with mp_pose.Pose(static_image_mode=False, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            H,W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # ===== วาดกรอบไกด์ (face-id style) =====
            box_h = int(H*0.9); box_w = int(W*0.5)
            x0 = (W - box_w)//2; y0 = (H - box_h)//2
            x1 = x0 + box_w;     y1 = y0 + box_h
            overlay = frame.copy()

            draw_guide_box(overlay, x0,y0,x1,y1)

            # เงื่อนไข: ต้องเห็นหัว-เท้าอยู่ในกรอบ (ด้วย landmark คร่าวๆ)
            # ===== เงื่อนไขท่าทาง / สเกลพิกเซลต่อ cm =====
            ok_pose = False
            px_per_cm = None
            lm = None
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                # ระยะหัว-เท้า (ถ้าเห็นเต็มตัว)
                fh_pix, head_y, foot_y = head_to_feet_pixels(lm, H)

                # ช่วงหัวไหล่-สะโพก (ใช้เป็นตัวแทนเมื่อเห็นแค่ครึ่งตัว)
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                sh_y = ((ls.y + rs.y)/2.0) * H
                hp_y = ((lh.y + rh.y)/2.0) * H
                torso_pix = max(0.0, hp_y - sh_y)

                # ให้อยู่กลางกรอบพอประมาณเหมือนเดิม
                midx = int(((ls.x+rs.x+lh.x+rh.x)/4.0)*W)
                box_h = int(H*0.9); box_w = int(W*0.5)
                x0 = (W - box_w)//2; x1 = x0 + box_w
                within_mid = (midx > x0+int(0.1*box_w)) and (midx < x1-int(0.1*box_w))

                # เกณฑ์ผ่านแบบเต็มตัว หรือแบบครึ่งตัว
                ok_full  = (fh_pix   > H*FULL_BODY_THRESH)
                ok_upper = (ALLOW_UPPER_BODY and torso_pix > H*0.18)   # ~18% ของความสูงภาพ

                if within_mid and (ok_full or ok_upper):
                    ok_pose = True
                    # คำนวณสเกลพิกเซล→ซม.
                    if ok_full:
                        px_per_cm = fh_pix / max(1e-6, height_cm)
                        pose_mode = "full"
                    else:
                        # ใช้สัดส่วนลำตัว ~30% ของส่วนสูงทั้งตัว
                        px_per_cm = torso_pix / max(1e-6, (TORSO_FRAC * height_cm))
                        pose_mode = "upper"
            else:
                pose_mode = "-"

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        print("❌ ไม่สามารถจับภาพนิ่งได้ ลองใหม่โดยยืนให้อยู่ในกรอบและนิ่งขึ้นอีกนิด"); sys.exit(3)

    frame_cap, lm_cap, px_per_cm = captured
    H,W = frame_cap.shape[:2]

    # ------- วัดคอ/เอว/สะโพก (มุมหน้าตรง) -------
    neck_y, waist_y, hip_y = levels_from_pose(lm_cap, H, W)
    bw = body_mask(frame_cap)
    neck_w_cm  = linescan_width(bw, neck_y)/px_per_cm
    waist_w_cm = linescan_width(bw, waist_y)/px_per_cm
    hip_w_cm   = linescan_width(bw, hip_y)/px_per_cm

    # ประมาณด้านข้างจากหน้าตรง
    neck_circ  = ramanujan_circumference(max(0.1,neck_w_cm/2.0),  max(0.1,(neck_w_cm*SIDE_SCALE)/2.0))
    waist_circ = ramanujan_circumference(max(0.1,waist_w_cm/2.0),  max(0.1,(waist_w_cm*SIDE_SCALE)/2.0))
    hip_circ   = ramanujan_circumference(max(0.1,hip_w_cm/2.0),    max(0.1,(hip_w_cm*SIDE_SCALE)/2.0))

    # ------- เพศ + ตัวชี้วัด -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, idx_to_class, img_size = load_gender_model(MODEL_PATH, device)
    gender, gconf = predict_gender(model, tfm, frame_cap, device, img_size, idx_to_class)

    bmi_val = weight_kg / ((height_cm/100.0)**2)
    bf_rfm  = rfm_percent(height_cm, max(1e-6, waist_circ), gender)
    bf_navy = navy_percent(height_cm, waist_circ, neck_circ, gender,
                        hip_circ_cm=(hip_circ if not gender.lower().startswith('m') else None))
    bmr = bmr_katch_mcardle(weight_kg, bf_rfm)

    # ===== Map เป็น “แบบคนทั่วไป”
    bf_label = bf_band_label(gender, bf_rfm)
    bf_ref   = bf_ref_nearest(gender, bf_rfm)  # ใกล้ภาพตัวอย่าง ~x%

    # ------- แสดงเส้นวัดบนภาพที่แคป + UI -------
    disp = frame_cap.copy()
    for y,name in [(neck_y,"Neck"),(waist_y,"Waist"),(hip_y,"Hip")]:
        y = int(round(y))
        cv2.line(disp, (0,y), (W-1,y), (0,255,0), 2)
        cv2.putText(disp, name, (10, max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # แถบหัว/ท้ายสรุปผล (อังกฤษเพื่อเลี่ยง ???)
    header = f"Gender: {gender} ({gconf*100:.1f}%)   BMI: {bmi_val:.2f}"
    footer = f"RFM: {bf_rfm:.1f}%  ≈ ~{bf_ref:.0f}%  [{bf_label}]   BMR: {bmr:.0f} kcal/day"
    draw_header_footer(disp, header, footer)

    cv2.imshow("Captured & Measured", disp)

    # ------- สรุปผล (คอนโซล ภาษาไทย) -------
    print("\n===== ผลลัพธ์ (นิ่งแล้วแคปอัตโนมัติ) =====")
    print(f"เพศ: {gender} ({gconf*100:.1f}%)")
    print(f"BMI: {bmi_val:.2f}")
    print(f"Neck circumference:  {neck_circ:.1f} cm")
    print(f"Waist circumference: {waist_circ:.1f} cm")
    print(f"Hip circumference:   {hip_circ:.1f} cm")
    print(f"Body Fat (RFM):      {bf_rfm:.1f}%  → หมวด: {bf_label}  (≈ ภาพตัวอย่าง ~{bf_ref:.0f}%)")
    if bf_navy is not None:
        print(f"Body Fat (U.S. Navy): {bf_navy:.1f}%")
    else:
        print("Body Fat (U.S. Navy): — (ข้อมูลไม่พอ/สมมติฐานไม่ครบ)")
    print(f"BMR (Katch–McArdle): {bmr:.0f} kcal/day")
    print(f"\nหมายเหตุ: มุมเดียวจึงประมาณด้านข้างด้วยสเกล {SIDE_SCALE:.2f}. "
        f"เพิ่มความแม่นยำได้โดยถ่ายด้านข้างเพิ่มอีกช็อตในอนาคต\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# ======== END ========