# MediaPipeHolistic

<p align="center">
  <b>Real-Time Computer Vision & AI</b>
  <br>
  Human pose tracking, object detection and camera-based AI with Python.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-111111?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/MediaPipe-ML%20Solutions-FF6F00?style=for-the-badge" alt="MediaPipe">
</p>

---

## 📌 About

**MediaPipeHolistic** is a computer vision project focused on real-time analysis using a camera.

The project combines modern AI and computer vision technologies such as **MediaPipe**, **YOLO**, **OpenCV** and **PyTorch** to experiment with real-time human and object understanding.

The main goal of the project is to explore how different computer vision models can be combined into a single real-time pipeline.

---

## ✨ Features

* 📷 **Real-time camera processing**
* 🧠 **AI-based object detection with YOLO**
* 🕺 **Human pose / holistic tracking with MediaPipe**
* ⚡ **Real-time computer vision pipeline**
* 🔬 Experiments with modern AI models
* 🖥️ Designed to run locally on Linux

---

## 🛠️ Tech Stack

| Technology    | Purpose                          |
| ------------- | -------------------------------- |
| **Python**    | Main programming language        |
| **MediaPipe** | Human landmark and pose tracking |
| **YOLO**      | Object detection                 |
| **OpenCV**    | Camera and image processing      |
| **PyTorch**   | Deep learning backend            |

---

## 📂 Project Structure

```text
MediaPipeHolistic/
│
├── main.py                  # Main application
├── test_owl.py              # Computer vision / model experiment
├── yolo11n.pt               # YOLO model weights
│
├── aegis_1787861642.png     # Project image
├── aegis_1787862743.png     # Project image
│
├── .gitignore               # Git ignored files
├── .python-version          # Python version configuration
└── README.md                # Project documentation
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/kr1on1x/MediaPipeHolistic.git
cd MediaPipeHolistic
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

If the project contains a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages manually according to the imports used by the project.

---

## ▶️ Running the Project

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
python main.py
```

The application uses the connected camera as the input source for real-time computer vision processing.

---

## 🤖 YOLO

The repository currently includes:

```text
yolo11n.pt
```

This is the YOLO model used for object detection experiments.

The `n` version is designed to provide a good balance between **inference speed and detection accuracy**, making it suitable for real-time applications.

---

## 📸 Computer Vision Pipeline

The general idea of the project is:

```text
                Camera
                   │
                   ▼
            ┌─────────────┐
            │   OpenCV    │
            │ Frame Input │
            └──────┬──────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌─────────────┐   ┌─────────────┐
   │  MediaPipe  │   │    YOLO     │
   │ Pose /      │   │   Object    │
   │ Landmarks   │   │  Detection  │
   └──────┬──────┘   └──────┬──────┘
          │                 │
          └────────┬────────┘
                   ▼
            Real-Time Analysis
```

This architecture makes it possible to combine **human understanding** and **object detection** in the same camera pipeline.

---

## 🧪 Experiments

This repository is also used as an experimental environment for computer vision and AI.

Possible directions include:

* Human pose analysis
* Object detection
* Gesture recognition
* Real-time interaction
* AI-assisted camera applications
* Combining multiple vision models
* Performance optimization

---

## 📈 Future Development

Planned improvements:

* [ ] Improve real-time inference performance
* [ ] Add more computer vision models
* [ ] Improve pose analysis
* [ ] Add configurable camera settings
* [ ] Add FPS and performance monitoring
* [ ] Improve project architecture
* [ ] Add a proper requirements file
* [ ] Add automated testing
* [ ] Add GPU acceleration where available
* [ ] Create a more advanced real-time AI pipeline

---

## 💻 Platform

The project is primarily developed and tested on:

```text
OS: Linux
Python: 3.x
Environment: Virtual Environment
```

GPU acceleration may be used when supported by the installed PyTorch / AI stack.

---

## 📜 License

This project is currently provided for educational and experimental purposes.

A dedicated open-source license can be added later.

---

## 👨‍💻 Author

**kr1on1x**

GitHub:
https://github.com/kr1on1x

---

<p align="center">
  <b>Computer Vision • Artificial Intelligence • Real-Time Systems</b>
</p>
