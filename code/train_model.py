# นำเข้า YOLO จาก Ultralytics
from ultralytics import YOLO

def main():
    # โหลดโมเดล YOLOv8
    model = YOLO('D:/Project/YoloV8/yolov8m.pt')  # ใช้ \\ หรือ / สำหรับ path

    # เทรนโมเดล
    model.train(
        data='data.yaml',        # ไฟล์ข้อมูลที่เตรียมไว้
        epochs=1000,            # จำนวนรอบการเทรน
        imgsz=640,               # ขนาดภาพที่ใช้
        batch=32,                # จำนวนข้อมูลต่อ batch
        patience=0,              # ไม่มี early stopping
        name='cow_top',  # ชื่อโฟลเดอร์บันทึกผลลัพธ์
        lr0=0.001,               # ค่า learning rate เริ่มต้น
        device="cuda"            # ใช้ GPU สำหรับการเทรน
    )

if __name__ == '__main__':
    main()