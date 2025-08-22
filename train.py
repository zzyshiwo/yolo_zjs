from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def main():
    print("Training YOLOv9 model...")
    model = YOLO("yolov9c.pt")
    results = model.train(data="coco8.yaml", epochs=2, imgsz=320, batch=1)

if __name__ == "__main__":
    main()  # 必须通过此结构启动