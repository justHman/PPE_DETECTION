import os
import cv2
from ultralytics import YOLO

from utils.caculator import inside
from utils.processor import get_color

# === Cấu hình các lớp ===
LABELS = {
    0: 'worker',
    1: 'helmet',
    2: 'vest',
    3: 'gloves',
    4: 'boots',
    5: 'no_helmet',
    6: 'no_vest',
    7: 'no_gloves',
    8: 'no_boots'
}

print("Các lớp phát hiện: ")
print("""
1: helmet
2: vest
3: gloves
4: boots
5: no_helmet
6: no_vest
7: no_gloves
8: no_boots
"""
)
ids = map(int, input("Nhập các lớp cần phát hiện (ví dụ: 1 2 3 cho helmet, vest, gloves): ").split())
REQUIRED_ITEMS = [LABELS[i] for i in ids if i in LABELS]
print(f"✅ Đã chọn các lớp: {REQUIRED_ITEMS}")

root = "weights\ppe"
MODELS = os.listdir(root)
print("Các model sẵn có: ")
for i, m in enumerate(MODELS):
    print(f"{i}: {m}")

model_id = MODELS[int(input("Chọn model bạn muốn sử dụng (nhập số tương ứng): "))]
model_path = os.path.join(root, model_id)
print(f"✅ Đã chọn model: {model_path}")


model = YOLO(model_path)  

media = input("Nhập đường dẫn video hoặc để trống để dùng webcam: ")
if media.strip() == "":
    media = 0
    
cap = cv2.VideoCapture(media)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Chạy YOLO detect trên frame ===
    results = model(frame, verbose=False, conf=0.2)[0]
    frame = results.plot()

    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    workers = []
    items = []

    # Gom nhóm worker & items
    for box, cls_id in zip(boxes, class_ids):
        label = names[cls_id]
        # if label.lower() != 'worker':
        #     x1, y1, x2, y2 = map(int, box)
        #     color = get_color(names[cls_id])
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(frame, label, (x1, y1 - 8),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label.lower() == 'worker':
            workers.append({'box': box, 'items': set()})
        elif label in LABELS.values():
            items.append({'box': box, 'label': label})

    # Kiểm tra trang bị của từng worker
    for worker in workers:
        wbox = worker['box']
        for item in items:
            if item['label'] in REQUIRED_ITEMS and inside(item['box'], wbox):
                worker['items'].add(item['label'])

    # Vẽ lên frame
    for worker in workers:
        x1, y1, x2, y2 = map(int, worker['box'])
        has_all = all(req in worker['items'] for req in REQUIRED_ITEMS) and \
                  all(f"no_{req}" not in worker['items'] for req in REQUIRED_ITEMS)
        color = (0, 255, 0) if has_all else (0, 0, 255)
        label_text = f"Worker ({'Safe' if has_all else 'Unsafe'})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("PPE Detection", frame)

    # Bấm Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Giải phóng tài nguyên ===
cap.release()
cv2.destroyAllWindows()
