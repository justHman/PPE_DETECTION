<div align="center">

# 🦺 Personal Protective Equipment (PPE) Detection System

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)

*A modern AI-powered system for detecting Personal Protective Equipment compliance in construction and industrial environments.*

[Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Documentation](#-documentation)

</div>

---

## 🎬 Demo

### Safe Worker Detection
Worker wearing **helmet** and **vest** ✅

*Result: **SAFE** - All required PPE items detected and no "no_*" items present*

![Safe Worker Demo](data/demos/SAFE.gif)

---

### Unsafe Worker Detection
Worker wearing **helmet** and **vest**, but **missing gloves** ❌

*Result: **UNSAFE** - Missing required PPE items or "no_*" items detected*

![Unsafe Worker Demo](data/demos/UNSAFE.gif)

---

## � Project Structure

```
PPE-Detection/
│
├── app/                          # Streamlit web application
│   ├── ui.py                     # Main UI interface
│   ├── backend.py                # Detection logic and processing
│   ├── style.css                 # Custom CSS styling
│   └── __pycache__/
│
├── config/                       # Configuration files
│   └── PPE_Dataset.yaml          # Dataset configuration
│
├── data/                         # Data directory
│   ├── demos/                    # Demo GIFs and media
│   │   ├── SAFE.gif              # Safe worker demo
│   │   └── UNSAFE.gif            # Unsafe worker demo
│   ├── images/                   # Image datasets
│   └── videos/                   # Video datasets
│
├── docs/                         # Documentation files
│
├── results/                      # Output directory for processed videos
│   └── ppe_detection_*.mp4       # Exported detection results
│
├── src/                          # Source modules
│   ├── creator.py                # Model creation utilities
│   └── model.py                  # Model architecture
│
├── utils/                        # Utility functions
│   ├── caculator.py              # Geometric calculations (IoU, inside check)
│   └── processor.py              # Data processing utilities
│
├── weights/                      # Model weights directory
│   ├── ppe/                      # PPE detection models
│   │   ├── ppe_8l_best.pt        # YOLOv8 Large model
│   │   ├── ppe_8s_best.pt        # YOLOv8 Small model
│   │   ├── ppe_rt_detr_best.pt   # RT-DETR model
│   │   └── ppe-8m.pt             # YOLOv8 Medium model
│   └── yolo/                     # Base YOLO weights
│
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🤖 Model Information

### Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | worker | Person/worker in the scene |
| 1 | helmet | Safety helmet worn |
| 2 | vest | Safety vest worn |
| 3 | gloves | Safety gloves worn |
| 4 | boots | Safety boots worn |
| 5 | no_helmet | Helmet not detected |
| 6 | no_vest | Vest not detected |
| 7 | no_gloves | Gloves not detected |
| 8 | no_boots | Boots not detected |

### Safety Classification Logic

A worker is classified as **SAFE** when:
- ✅ All selected PPE items are detected within the worker's bounding box
- ✅ None of the "no_*" items related to the required PPE are detected

A worker is classified as **UNSAFE** when:
- ❌ One or more required PPE items are missing
- ❌ "no_*" items related to the required PPE are detected

---

##  Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Clone Repository](#clone-repository)
  - [Install Dependencies](#install-dependencies)
- [Usage](#-usage)
  - [Web Application (Streamlit)](#web-application-streamlit)
  - [Command Line Interface](#command-line-interface)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Development](#️-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

The **PPE Detection System** is an intelligent real-time monitoring solution that uses advanced computer vision and deep learning to ensure workplace safety compliance. Built on **YOLOv8** architecture and powered by a modern **Streamlit** interface, the system automatically detects and verifies the presence of essential safety equipment on workers.

### Key Capabilities:
- ✅ **Real-time Detection**: Process live camera feeds or video files
- 🎯 **Multi-PPE Recognition**: Detect helmets, vests, gloves, and boots
- 🚦 **Safety Compliance**: Automatic worker status classification (Safe/Unsafe)
- 📊 **Intuitive Dashboard**: Modern web interface with customizable settings
- 💾 **Export Results**: Save annotated videos with detection results
- ⚡ **High Performance**: Optimized for speed and accuracy

---

## 🚀 Installation

### Prerequisites

Before installation, ensure you have:

- **Python 3.10+** installed ([Download Python](https://www.python.org/downloads/))
- **Git** for cloning the repository ([Download Git](https://git-scm.com/downloads))
- (Optional) **CUDA-enabled GPU** for faster inference
- **Webcam** (optional, for live detection)

### Clone Repository

```bash
# Clone the repository
git clone https://github.com/justHman/PPE-Detection.git

# Navigate to project directory
cd PPE-Detection
```

### Install Dependencies

#### Option 1: Using pip (Recommended)

```bash
# Install required packages
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n ppe-detection python=3.10

# Activate environment
conda activate ppe-detection

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

Place your trained YOLOv8 model weights (`.pt` files) in the `weights/ppe/` directory:

```
weights/
└── ppe/
    ├── ppe_8l_best.pt
    ├── ppe_8s_best.pt
    ├── ppe_rt_detr_best.pt
    └── ppe-8m.pt
```

---

## 💻 Usage

### Web Application (Streamlit)

Launch the modern web interface:

```bash
# Navigate to project root
cd PPE-Detection

# Run Streamlit app
streamlit run app/ui.py
```

The application will open in your default browser at `http://localhost:8501`

#### Using the Web Interface:

1. **Select Model**: Choose a YOLOv8 model from the sidebar
2. **Choose PPE Items**: Select which equipment to detect (helmet, vest, gloves, boots)
3. **Adjust Confidence**: Set detection confidence threshold (0.1 - 1.0)
4. **Configure Export** (optional): Enable video export and set output path
5. **Select Input Source**:
   - **Upload Video**: Browse and upload a video file
   - **Enter Path**: Specify path to a local video file
   - **Use Camera**: Select camera ID for live detection
6. **Start Detection**: Click the "🚀 Bắt đầu phát hiện" button
7. **Monitor Results**: View real-time detection with FPS counter
8. **Download Results**: After completion, download the annotated video

---

### Command Line Interface

For terminal-based detection:

```bash
# Run the CLI version
python main.py
```

Follow the interactive prompts:
1. Select PPE items to detect
2. Choose a model
3. Enter video path or press Enter for webcam
4. Press 'Q' to stop detection

---

## ⚙️ Configuration

### Dataset Configuration

Edit `config/PPE_Dataset.yaml` to configure your dataset:

```yaml
path: /path/to/dataset          # Dataset root directory
train: images/train             # Training images
val: images/valid               # Validation images
test: images/test               # Test images (optional)

# Classes
names:
  0: worker
  1: helmet
  2: vest
  3: gloves
  4: boots
  5: no_helmet
  6: no_vest
  7: no_gloves
  8: no_boots
```

### Model Configuration

Available models in `weights/ppe/`:
- **ppe_8s_best.pt**: Fastest, suitable for real-time applications
- **ppe-8m.pt**: Balanced speed and accuracy
- **ppe_8l_best.pt**: Highest accuracy, slower inference
- **ppe_rt_detr_best.pt**: Alternative RT-DETR architecture

---

## 📚 Documentation

### Key Modules

#### `app/backend.py`
Core detection logic:
- `PPEDetector`: Main detection class
- `run_detection()`: Generator for frame-by-frame processing
- `get_available_models()`: List available model weights
- `get_all_ppe_labels()`: Return PPE class labels

#### `app/ui.py`
Streamlit interface:
- Interactive sidebar controls
- Real-time video streaming
- Export functionality
- Session state management

#### `utils/caculator.py`
Geometric utilities:
- `inside()`: Check if bounding box A is inside box B

#### `utils/processor.py`
Processing utilities:
- `get_color()`: Get color coding for each PPE class
- `yolo_type()`: Get YOLO model parameters

---

## 🛠️ Development

### Adding New Models

1. Train your YOLOv8 model with the PPE dataset
2. Save the best weights (`.pt` file)
3. Copy to `weights/ppe/` directory
4. The model will automatically appear in the UI dropdown

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/justHman/PPE-Detection.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection framework
- **[Streamlit](https://streamlit.io/)** - Beautiful web application framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **PPE Dataset Contributors** - For providing training data

---

## 📧 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/justHman/PPE-Detection/issues)
- **Email**: hmanclubs11@gmail.com

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Built with ❤️ for workplace safety**

[Back to Top](#-personal-protective-equipment-ppe-detection-system)

</div>
