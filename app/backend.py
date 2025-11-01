"""
Backend module cho PPE Detection
Chứa toàn bộ logic xử lý từ main.py
"""

import os
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Import utils từ thư mục gốc
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.caculator import inside
from utils.processor import get_color


class PPEDetector:
    """Class quản lý PPE detection với YOLO model"""
    
    # Định nghĩa các lớp PPE
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
    
    def __init__(self, model_path, required_items, conf_threshold=0.5):
        """
        Khởi tạo PPE Detector
        
        Args:
            model_path (str): Đường dẫn đến model .pt
            required_items (list): Danh sách các PPE cần phát hiện
            conf_threshold (float): Ngưỡng confidence
        """
        self.model_path = model_path
        self.required_items = required_items
        self.conf_threshold = conf_threshold
        self.model = None
        self.fps = 0
        
    def load_model(self):
        """Load YOLO model từ file .pt"""
        if self.model is None:
            self.model = YOLO(self.model_path)
            self.model.conf = self.conf_threshold
        return self.model
    
    def process_frame(self, frame):
        start_time = time.time()
        
        # Chạy YOLO detection
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        
        # Lấy thông tin detection
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        names = results.names
        
        workers = []
        items = []
        
        # Phân loại workers và PPE items
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            label = names[cls_id]
            
            if label.lower() == 'worker':
                workers.append({'box': box, 'items': set(), 'conf': conf})
            elif label in self.LABELS.values():
                items.append({'box': box, 'label': label, 'conf': conf})
        
        # Kiểm tra trang bị của từng worker
        for worker in workers:
            wbox = worker['box']
            for item in items:
                if item['label'] in self.required_items and inside(item['box'], wbox):
                    worker['items'].add(item['label'])
        
        # Vẽ PPE items trước
        # frame = results.plot()
        for item in items:
            x1, y1, x2, y2 = map(int, item['box'])
            color = get_color(item['label'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label_text = f"{item['label']} {item['conf']:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Vẽ workers với màu phù hợp
        for worker in workers:
            x1, y1, x2, y2 = map(int, worker['box'])
            has_all = all(req in worker['items'] for req in self.required_items) and \
                      all(f"no_{req}" not in worker['items'] for req in self.required_items)
            
            # Safe: xanh lá, Unsafe: đỏ
            color = (0, 255, 0) if has_all else (0, 0, 255)
            status = 'Safe' if has_all else 'Unsafe'
            label_text = f"Worker ({status}) {worker['conf']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Hiển thị missing items nếu unsafe
            if not has_all:
                missing = set(self.required_items) - worker['items']
                missing_text = f"Missing: {', '.join(missing)}"
                cv2.putText(frame, missing_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Tính FPS
        end_time = time.time()
        self.fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        return frame, self.fps


def get_available_models(weights_dir="weights/ppe"):
    """
    Lấy danh sách các model .pt có sẵn
    
    Args:
        weights_dir (str): Thư mục chứa model weights
        
    Returns:
        list: Danh sách các file .pt
    """
    root_path = Path(__file__).parent.parent / weights_dir
    if not root_path.exists():
        return []
    
    models = [f for f in os.listdir(root_path) if f.endswith('.pt')]
    return sorted(models)


def run_detection(model_path, required_items, conf_threshold, source, stop_flag=None, export_path=None):
    """
    Generator function để chạy detection và yield frame từng bước
    
    Args:
        model_path (str): Đường dẫn đến model
        required_items (list): Danh sách PPE cần detect
        conf_threshold (float): Ngưỡng confidence
        source: Nguồn video (0 cho webcam, path cho video file, hoặc uploaded file)
        stop_flag (function): Hàm callback để kiểm tra có dừng không
        export_path (str): Đường dẫn để lưu video kết quả (None = không lưu)
        
    Yields:
        tuple: (frame, fps)
    """
    # Khởi tạo detector
    detector = PPEDetector(model_path, required_items, conf_threshold)
    detector.load_model()
    
    tmp_path = None
    cap = None
    video_writer = None
    
    try:
        # Mở video capture
        if isinstance(source, int):
            # Webcam
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # Sử dụng DirectShow trên Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)  # Fallback
            
            if not cap.isOpened():
                raise ValueError(f"Không thể mở camera ID {source}. Vui lòng kiểm tra:\n"
                               f"- Camera đã được kết nối chưa?\n"
                               f"- Ứng dụng khác có đang sử dụng camera không?")
        
        elif isinstance(source, str):
            # File path
            if not Path(source).exists():
                raise ValueError(f"File không tồn tại: {source}")
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ValueError(f"Không thể mở file video: {source}\n"
                               f"- Định dạng file có được hỗ trợ không?\n"
                               f"- File có bị lỗi không?")
        
        else:
            # Uploaded file (bytes)
            import tempfile
            try:
                # Lấy tên file gốc nếu có
                file_ext = '.mp4'
                if hasattr(source, 'name'):
                    file_ext = Path(source.name).suffix or '.mp4'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(source.read())
                    tmp_path = tmp_file.name
                
                cap = cv2.VideoCapture(tmp_path)
                
                if not cap.isOpened():
                    raise ValueError(f"Không thể mở file video đã upload.\n"
                                   f"- Định dạng file: {file_ext}\n"
                                   f"- Vui lòng thử upload file khác hoặc định dạng khác (MP4, AVI, MOV)")
            
            except Exception as e:
                if tmp_path and Path(tmp_path).exists():
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                raise ValueError(f"Lỗi khi xử lý file upload: {str(e)}")
        
        # Kiểm tra cuối cùng
        if not cap or not cap.isOpened():
            raise ValueError("Không thể mở nguồn video. Vui lòng thử lại.")
        
        # Thiết lập video writer nếu cần export
        if export_path:
            # Tạo thư mục nếu chưa có
            export_dir = Path(export_path).parent
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Lấy thông tin video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            
            # Khởi tạo VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(export_path),
                fourcc,
                fps_video,
                (frame_width, frame_height)
            )
        
        # Đọc và xử lý frames
        frame_count = 0
        while cap.isOpened():
            # Kiểm tra stop flag
            if stop_flag and stop_flag():
                break
            
            ret, frame = cap.read()
            if not ret:
                if frame_count == 0:
                    raise ValueError("Không thể đọc frame từ video. File có thể bị lỗi.")
                break
            
            # Process frame
            processed_frame, fps = detector.process_frame(frame)
            
            # Ghi frame vào video nếu có export
            if video_writer is not None:
                video_writer.write(processed_frame)
            
            # Convert BGR to RGB cho Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            yield processed_frame_rgb, fps
            
    except Exception as e:
        # Re-raise với thông tin chi tiết
        raise
        
    finally:
        # Giải phóng resources
        if cap is not None:
            cap.release()
        
        if video_writer is not None:
            video_writer.release()
        
        # Xóa temp file nếu có
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except:
                pass


def get_all_ppe_labels():
    """
    Lấy danh sách tất cả các label PPE (trừ worker)
    
    Returns:
        list: Danh sách các label PPE
    """
    # Chỉ trả về các PPE chính (không bao gồm no_*)
    labels = ['helmet', 'vest', 'gloves', 'boots']
    return labels
