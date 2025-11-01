"""
🦺 Personal Protective Equipment (PPE) Detection
Ứng dụng Streamlit hiện đại để phát hiện đồ bảo hộ lao động
"""

import streamlit as st
import cv2
from pathlib import Path
from datetime import datetime
import sys

# Import backend module
from backend import (
    PPEDetector,
    get_available_models,
    run_detection,
    get_all_ppe_labels
)

# ============ Cấu hình trang ============
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Load CSS ============
def load_css():
    """Load custom CSS từ file style.css"""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============ Session State ============
if 'detecting' not in st.session_state:
    st.session_state.detecting = False
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False

# ============ Header ============
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        ---Personal Protective Equipment Detection---
    </h1>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #7f8c9a; font-size: 1.1rem; margin-top: 0.5rem;'>Hệ thống phát hiện đồ bảo hộ lao động thông minh với AI</p>", unsafe_allow_html=True)

st.divider()

# ============ Sidebar - Cấu hình ============
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình hệ thống")
    
    # === Model Selection ===
    st.markdown("### 🤖 Chọn Model")
    available_models = get_available_models()
    
    if not available_models:
        st.error("⚠️ Không tìm thấy model trong thư mục weights/ppe/")
        st.stop()
    
    selected_model = st.selectbox(
        "Model",
        available_models,
        help="Chọn model YOLO đã train để sử dụng"
    )
    
    model_path = Path(__file__).parent.parent / "weights" / "ppe" / selected_model
    
    st.divider()
    
    # === Label Selection ===
    st.markdown("### 🏷️ Chọn PPE cần phát hiện")
    all_labels = get_all_ppe_labels()
    
    # Tạo checkbox cho từng label
    selected_labels = []
    for label in all_labels:
        if st.checkbox(label, value=True, key=f"label_{label}"):
            selected_labels.append(label)
    
    if not selected_labels:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 loại PPE")
    
    st.divider()
    
    # === Confidence Threshold ===
    st.markdown("### 🎯 Confidence Threshold")
    confidence = st.slider(
        "Ngưỡng tin cậy",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Chỉ hiển thị detection có confidence >= ngưỡng này"
    )
    
    st.caption(f"Ngưỡng hiện tại: **{confidence:.2f}**")
    
    st.divider()
    
    # === Export Settings ===
    st.markdown("### 💾 Xuất kết quả")
    export_video = st.checkbox(
        "Lưu video kết quả",
        value=False,
        help="Xuất video đã detect ra file"
    )
    
    export_path = None
    if export_video:
        use_custom_path = st.checkbox("Tùy chỉnh đường dẫn", value=False)
        
        if use_custom_path:
            custom_path = st.text_input(
                "Đường dẫn lưu file",
                placeholder="VD: D:\\Videos\\output.mp4",
                help="Nhập đường dẫn đầy đủ để lưu video"
            )
            if custom_path:
                export_path = custom_path
        else:
            # Sử dụng thư mục results mặc định
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(__file__).parent.parent / "results"
            export_path = str(results_dir / f"ppe_detection_{timestamp}.mp4")
            st.info(f"📁 Sẽ lưu tại: `{export_path}`")
    
    st.divider()
    
    # === Input Source ===
    st.markdown("### 📹 Nguồn đầu vào")
    source_type = st.radio(
        "Chọn nguồn video",
        ["📤 Upload video", "📂 Nhập đường dẫn", "📷 Sử dụng camera"],
        help="Chọn nguồn video để phát hiện PPE"
    )
    
    video_source = None
    
    if source_type == "📤 Upload video":
        uploaded_file = st.file_uploader(
            "Chọn file video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload video để phát hiện PPE"
        )
        if uploaded_file:
            video_source = uploaded_file
            st.success(f"✅ Đã chọn: {uploaded_file.name}")
    
    elif source_type == "📂 Nhập đường dẫn":
        video_path = st.text_input(
            "Đường dẫn video",
            placeholder="VD: D:\\Videos\\construction.mp4",
            help="Nhập đường dẫn đầy đủ đến file video"
        )
        if video_path:
            if Path(video_path).exists():
                video_source = video_path
                st.success(f"✅ Đường dẫn hợp lệ")
            else:
                st.error("❌ File không tồn tại")
    
    elif source_type == "📷 Sử dụng camera":
        camera_id = st.number_input(
            "Camera ID",
            min_value=0,
            max_value=10,
            value=0,
            help="ID của camera (thường là 0 cho webcam mặc định)"
        )
        video_source = int(camera_id)
        st.info(f"📷 Sẽ sử dụng Camera ID: {camera_id}")
    
    st.divider()
    
    # === Start Detection Button ===
    # Kiểm tra điều kiện để enable button
    can_start = True
    warning_msg = None
    
    if not selected_labels:
        can_start = False
        warning_msg = "⚠️ Chọn ít nhất 1 PPE để bắt đầu"
    elif video_source is None:
        can_start = False
        if source_type == "� Upload video":
            warning_msg = "⚠️ Vui lòng upload file video"
        elif source_type == "📂 Nhập đường dẫn":
            warning_msg = "⚠️ Vui lòng nhập đường dẫn video hợp lệ"
        else:
            warning_msg = "⚠️ Chọn nguồn video để bắt đầu"
    
    if can_start:
        if st.button("🚀 Bắt đầu phát hiện", key="start_btn"):
            st.session_state.detecting = True
            st.session_state.stop_detection = False
            st.rerun()
    else:
        st.button("🚀 Bắt đầu phát hiện", disabled=True)
        if warning_msg:
            st.warning(warning_msg)

# ============ Main Area ============
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.session_state.detecting:
        st.markdown("### 📺 Video Stream")
        
        # Stop button
        if st.button("⏹️ Dừng phát hiện", key="stop_btn"):
            st.session_state.detecting = False
            st.session_state.stop_detection = True
            st.rerun()
        
        # Placeholder cho video stream
        video_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        # Hiển thị thông tin export nếu có
        if export_video and export_path:
            st.info(f"💾 Đang ghi video vào: `{export_path}`")
        
        try:
            # Chạy detection
            frame_count = 0
            for frame, fps in run_detection(
                model_path=str(model_path),
                required_items=selected_labels,
                conf_threshold=confidence,
                source=video_source,
                stop_flag=lambda: st.session_state.stop_detection,
                export_path=export_path if export_video else None
            ):
                # Hiển thị frame
                video_placeholder.image(
                    frame,
                    channels="RGB",
                    width="stretch",
                    # caption=f"PPE Detection - Frame {frame_count}"
                )
                
                # Hiển thị FPS
                fps_placeholder.markdown(
                    f"<p style='text-align: center; color: #4da6ff; font-size: 1.2rem;'>"
                    f"⚡ FPS: <strong>{fps:.1f}</strong></p>",
                    unsafe_allow_html=True
                )
                
                frame_count += 1
                
                # Kiểm tra stop flag
                if st.session_state.stop_detection:
                    break
            
            # Kết thúc detection
            st.session_state.detecting = False
            st.success("✅ Đã hoàn thành phát hiện!")
            
            # Thông báo nếu đã lưu video
            if export_video and export_path:
                if Path(export_path).exists():
                    file_size = Path(export_path).stat().st_size / (1024 * 1024)  # MB
                    st.success(f"💾 Video đã được lưu: `{export_path}` ({file_size:.2f} MB)")
                    
                    # Download button
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            label="📥 Tải video xuống",
                            data=f,
                            file_name=Path(export_path).name,
                            mime="video/mp4"
                        )
                else:
                    st.warning("⚠️ Không thể lưu video")
            
            if st.button("🔄 Phát hiện lại"):
                st.session_state.detecting = True
                st.session_state.stop_detection = False
                st.rerun()
        
        except Exception as e:
            st.session_state.detecting = False
            st.error(f"❌ Lỗi: {str(e)}")
            st.exception(e)
    
    else:
        # Hiển thị hướng dẫn khi chưa bắt đầu
        st.markdown("""
            <div style='text-align: center; padding: 3rem; background-color: #1a2332; border-radius: 12px; border: 2px dashed #2a3f5f;'>
                <h2 style='color: #66d9ff; margin-bottom: 1rem;'>👈 Cấu hình bên trái</h2>
                <p style='color: #99ccff; font-size: 1.1rem; line-height: 1.8;'>
                    1️⃣ Chọn model YOLO<br>
                    2️⃣ Chọn các loại PPE cần phát hiện<br>
                    3️⃣ Điều chỉnh confidence threshold<br>
                    4️⃣ Chọn nguồn video (Upload/Path/Camera)<br>
                    5️⃣ Nhấn <strong style='color: #4da6ff;'>"Bắt đầu phát hiện"</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

# ============ Footer với thông tin ============
st.divider()

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
        <div style='background-color: #1a2332; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #66d9ff; margin-bottom: 0.5rem;'>📊 Model</h4>
            <p style='color: #e0e6ed;'>{}</p>
        </div>
    """.format(selected_model if selected_model else "Chưa chọn"), unsafe_allow_html=True)

with col_info2:
    st.markdown("""
        <div style='background-color: #1a2332; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #66d9ff; margin-bottom: 0.5rem;'>🏷️ PPE Items</h4>
            <p style='color: #e0e6ed;'>{}</p>
        </div>
    """.format(len(selected_labels) if selected_labels else 0), unsafe_allow_html=True)

with col_info3:
    st.markdown("""
        <div style='background-color: #1a2332; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #66d9ff; margin-bottom: 0.5rem;'>🎯 Confidence</h4>
            <p style='color: #e0e6ed;'>{:.0%}</p>
        </div>
    """.format(confidence), unsafe_allow_html=True)

# Footer text
st.markdown("""
    <footer style='margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #2a3f5f;'>
        <p style='text-align: center; color: #7f8c9a;'>
            🦺 PPE Detection System | Powered by YOLO & Streamlit | 
            <a href='https://github.com/justHman/PPE_DETECTION' target='_blank' style='color: #4da6ff;'>GitHub</a>
        </p>
    </footer>
""", unsafe_allow_html=True)
