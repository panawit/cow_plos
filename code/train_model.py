# MIT License

# Copyright (c) 2025 Panawit Hanpinitsak

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
