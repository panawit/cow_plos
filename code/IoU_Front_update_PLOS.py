import csv
import os
from datetime import timedelta
from ultralytics import YOLO
import cv2
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

# ใช้ backend non-interactive เพื่อลดข้อผิดพลาดเกี่ยวกับ tkinter
matplotlib.use('Agg')
iou_list = []

# Load the top models
front_act_model = YOLO("D:\\Project\\YoloV8\\cow\\Cow_act_front_new_best.pt")
front_iden_model = YOLO("D:\Project\YoloV8\cow\Cow_front_name.pt")

iou_threshold_duplicate = 0.8
iou_threshold_mapping = 0.5

# Video input
front_source = "D:\Project\YoloV8\cow\A04_20240517064400.mp4"
fileName = os.path.basename(front_source)
camera = fileName[0:3]
date = fileName[4:12]
hour_start = int(fileName[12:14])
minute_start = int(fileName[14:16])
second_start = int(fileName[16:18])
total_second = hour_start * 3600 + minute_start * 60 + second_start

date_obj = datetime.strptime(date, "%Y%m%d")
date = date_obj.strftime("%d-%m-%Y")
print(date)
print(hour_start)
print(minute_start)
print(second_start)

def convert_total_to_hms(total_second):
    # แปลง total_second เป็นชั่วโมง, นาที และวินาที
    hour = int(total_second // 3600)
    min = int((total_second % 3600) // 60)
    sec = int(total_second % 60)

    # ตรวจสอบว่าต้องเปลี่ยนวันที่หรือไม่
    flag_date_change = total_second >= 86400  # 1 วันมี 86400 วินาที
    if flag_date_change:
        total_second %= 86400  # รีเซ็ต total_second ให้เริ่มต้นใหม่หลังครบ 1 วัน

    return hour, min, sec, total_second, flag_date_change

def process_video_duration(video_path):
    """Get total duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return int(total_frames / fps)

video_duration = process_video_duration(front_source)
max_seconds = video_duration + total_second

def process_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return int(total_frames / fps)

video_duration = process_video_duration(front_source)
max_seconds = video_duration + total_second

def center_crop_frame(frame):
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    return frame[start_h:start_h + min_dim, start_w:start_w + min_dim]


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    return inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

def filter_duplicates(detections, iou_threshold=0.5):
    filtered = []
    for i, det in enumerate(detections):
        if not any(calculate_iou(det['box'], detections[j]['box']) > iou_threshold for j in range(i)):
            filtered.append(det)
    return filtered


def plot_bounding_boxes(frame, iden_detections, act_detections):
    frame = center_crop_frame(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.imshow(frame_rgb)
    import matplotlib.patches as patches
    for det in iden_detections:
        x1, y1, x2, y2 = det['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1-10, det['label'], color='red', fontsize=10, weight='bold')
    for det in act_detections:
        x1, y1, x2, y2 = det['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1-10, det['label'], color='blue', fontsize=10, weight='bold')
    plt.close(fig)

# Cow label mapping
iou_list = []
cow_patterns = {
    'cow-a-black': 'Cow_A_Activity',
    'cow-b-white-pattern': 'Cow_B_Activity',
    'cow-c-black-pattern': 'Cow_C_Activity'
}

# Matching and processing

def match_results(iden_result, act_result, total_second, date, last_activities=None, frame=None):
    hour, min, sec, total_second, flag_date_change = convert_total_to_hms(total_second)
    time_str = f"{hour:02d}:{min:02d}:{sec:02d}"
    if flag_date_change:
        date = (datetime.strptime(date, "%d-%m-%Y") + timedelta(days=1)).strftime("%d-%m-%Y")
    matches = []
    iden_detections = [
        {'label': iden.names[int(cls.item())], 'box': box.tolist(), 'confidence': conf.item(), 'frame': iden.orig_shape}
        for iden in iden_result for box, conf, cls in zip(iden.boxes.xyxy, iden.boxes.conf, iden.boxes.cls)
    ]
    act_detections = [
        {'label': act.names[int(cls.item())], 'box': box.tolist(), 'confidence': conf.item(), 'frame': act.orig_shape}
        for act in act_result for box, conf, cls in zip(act.boxes.xyxy, act.boxes.conf, act.boxes.cls)
    ]
    iden_detections = filter_duplicates(iden_detections, iou_threshold_duplicate)
    act_detections = filter_duplicates(act_detections, iou_threshold_duplicate)
    if last_activities is None:
        last_activities = {v:('stand',0.0) for v in cow_patterns.values()}
    detected = {d['label'] for d in iden_detections}
    # compute IOUs
    for iden in iden_detections:
        for act in act_detections:
            if iden['frame']==act['frame']:
                iou = calculate_iou(iden['box'], act['box'])
                iou_list.append(iou)
    # match
    for iden in iden_detections:
        matched=False
        for act in act_detections:
            if iden['frame']==act['frame']:
                iou = calculate_iou(iden['box'], act['box'])
                if iou>iou_threshold_mapping:
                    matches.append((iden['label'], act['label'], iou, act['confidence'], iden['confidence'], date, time_str))
                    last_activities[cow_patterns[iden['label']]] = (act['label'], act['confidence'])
                    matched=True
                    break
        if not matched:
            key=cow_patterns[iden['label']]
            matches.append((iden['label'], last_activities[key][0], 0.0, last_activities[key][1], iden['confidence'], date, time_str))
    for lbl,key in cow_patterns.items():
        if lbl not in detected:
            matches.append((lbl, last_activities[key][0], -1.0, last_activities[key][1], 0.0, date, time_str))
    if frame is not None:
        plot_bounding_boxes(frame, iden_detections, act_detections)
    return matches, last_activities, total_second, date

# Video processing
def process_video(iden_model, act_model, source, total_second, date):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    matches=[]
    last_acts=None
    frame_num=0
    while frame_num<total_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = center_crop_frame(frame)
        idr = iden_model.predict(frame, conf=0.5)
        acr = act_model.predict(frame, conf=0.5)
        m, last_acts, total_second, date = match_results(idr, acr, total_second, date, last_activities=last_acts, frame=frame)
        matches.extend(m)
        plt.close('all')
        frame_num+=1
        total_second+=1/fps
    cap.release()
    return matches, last_acts

print("Processing top video...")
top_matches, top_last_activities = process_video(front_iden_model, front_act_model, front_source, total_second, date)

# Map for CSV
cow_mapping = {
    'cow-a-black':'cow-a (black)',
    'cow-b-white-pattern':'cow-b (white-pattern)',
    'cow-c-black-pattern':'cow-c (black-pattern)'
}
# Define CSV columns including IOU
csv_columns = [
    'Date','Time',
    'cow-a (black)','cow-a (black) (detection confidence)','cow-a (black) (Activity confidence)','cow-a (black) (iou)',
    'cow-b (white-pattern)','cow-b (white-pattern) (detection confidence)','cow-b (white-pattern) (Activity confidence)','cow-b (white-pattern) (iou)',
    'cow-c (black-pattern)','cow-c (black-pattern) (detection confidence)','cow-c (black-pattern) (Activity confidence)','cow-c (black-pattern) (iou)'
]

# Build behavior table
behavior_table=[]
for match in top_matches:
    cow_label, activity, iou, act_conf, det_conf, date, time = match
    formatted_time = time.replace(':','.')
    row = next((r for r in behavior_table if r['Date']==date and r['Time']==formatted_time), None)
    if not row:
        row = {'Date':date,'Time':formatted_time}
        # init empty fields
        for col in csv_columns[2:]: row[col] = ''
        behavior_table.append(row)
    base = cow_mapping[cow_label]
    row[base] = activity
    row[f"{base} (detection confidence)"] = det_conf
    row[f"{base} (Activity confidence)"] = act_conf
    row[f"{base} (iou)"] = iou

# Write to CSV
csv_file = (
    f"D:\\Project\\YoloV8\\cow\\cow_behavior_front_IoU_{fileName[4:18]}_{iou_threshold_duplicate}_{iou_threshold_mapping}_Jess.csv"
)
try:
    with open(csv_file,'w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(behavior_table)
    print(f"CSV file {csv_file} created successfully.")
except IOError:
    print("I/O error occurred")

# Plot IOU histogram
sns.histplot(iou_list)
file_name = (
    f"D:\\Project\\YoloV8\\cow\\cow_behavior_front_iou_histogram_{fileName[4:18]}_{iou_threshold_duplicate}_{iou_threshold_mapping}.png"
)
plt.savefig(file_name,dpi=300,bbox_inches='tight')
