# AEGIS — Multimodal Vision Core

<p align="center">
  <b>Real-Time Multimodal Computer Vision System</b>
  <br>
  <sub>MediaPipe • YOLO11 • OWLv2 • OpenCV • Gesture Control</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/YOLO11-Object%20Detection-111111?style=for-the-badge">
  <img src="https://img.shields.io/badge/MediaPipe-Holistic-FF6F00?style=for-the-badge">
  <img src="https://img.shields.io/badge/OWLv2-Open%20Vocabulary-8A2BE2?style=for-the-badge">
</p>

---

## 🧠 Overview

**AEGIS** is a real-time multimodal computer vision system that combines several AI models into a single camera-based interface.

The system processes a live camera feed and combines:

* **MediaPipe Holistic** for face and hand landmark tracking
* **YOLO11n** for real-time object detection
* **OWLv2** for open-vocabulary object search
* **OpenCV** for image processing and real-time visualization
* **Hand gestures** for interacting with detected objects

The project is designed as an experimental platform for building interactive AI vision systems.

---

## ⚡ Core Features

### 👤 Human Tracking

AEGIS uses MediaPipe Holistic to track:

* Face landmarks
* Left hand
* Right hand
* Hand positions
* Hand gestures

The face is rendered as a real-time landmark mesh.

---

### 🖐️ Gesture Control

The system recognizes several gestures:

| Gesture           | Action                       |
| ----------------- | ---------------------------- |
| ✌️ **V SIGN**     | Switch operating mode        |
| 🤏 **PINCH**      | Select / lock onto an object |
| 🖐️ **OPEN HAND** | Release target               |
| 👍 **THUMBS UP**  | Confirm selected target      |

Gestures are processed directly from hand landmarks.

---

### 🎯 Object Detection

AEGIS uses **YOLO11n** for real-time object detection.

Detected objects receive:

* Bounding boxes
* Object labels
* Confidence scores
* Object IDs
* Center points
* Object-specific visualization

The system also keeps track of:

```text
PEOPLE
ANIMALS
OBJECTS
```

---

### 🔎 Open-Vocabulary Search

Unlike traditional object detection models with a fixed list of classes, AEGIS also includes **OWLv2**.

This allows the system to search for objects using a natural-language query.

For example:

```text
SEARCH: red backpack
```

or:

```text
SEARCH: laptop
```

The query is processed by:

```text
google/owlv2-base-patch16-ensemble
```

This makes the system capable of experimenting with **open-vocabulary object detection**.

---

## 🎯 Target Lock System

AEGIS combines hand tracking with object detection to create an interactive target-selection system.

The general interaction is:

```text
Hand
 │
 ▼
Pointer
 │
 ▼
Object Detection
 │
 ▼
Object Under Pointer
 │
 ▼
PINCH
 │
 ▼
TARGET LOCK
 │
 ▼
THUMBS UP
 │
 ▼
TARGET CONFIRMED
```

The active target is highlighted using the interface's targeting visualization.

---

## 🖥️ Interface

AEGIS includes a custom real-time HUD inspired by futuristic computer interfaces.

The interface displays:

```text
AEGIS
MULTIMODAL VISION CORE

ONLINE
FPS
MODE

BIOMETRIC
FACE
R-HAND
L-HAND

ENVIRONMENT
PEOPLE
ANIMALS
OBJECTS
```

The interface also provides:

* Real-time FPS
* Current operating mode
* Hand gesture status
* Face tracking status
* Object statistics
* Search interface
* Target lock panel
* Camera controls

---

## 🔄 System Architecture

```text
                         ┌──────────────────┐
                         │      CAMERA      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      OpenCV      │
                         │   Frame Input    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐         ┌─────────────────┐
           │    MediaPipe    │         │      YOLO11     │
           │    Holistic     │         │     Object      │
           │                 │         │    Detection    │
           │ Face + Hands    │         └────────┬────────┘
           └────────┬────────┘                  │
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Gesture / Pointer│
                         │    Controller    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Target Lock   │
                         └──────────────────┘


                  Open Vocabulary Search
                           │
                           ▼
                    ┌──────────────┐
                    │    OWLv2     │
                    │ Text Query → │
                    │    Objects   │
                    └──────────────┘
```

---

## 🛠️ Technologies

| Technology       | Role                                   |
| ---------------- | -------------------------------------- |
| **Python**       | Main development language              |
| **OpenCV**       | Camera, image processing and rendering |
| **MediaPipe**    | Face and hand landmark tracking        |
| **YOLO11n**      | Real-time object detection             |
| **OWLv2**        | Open-vocabulary object detection       |
| **PyTorch**      | Deep learning inference                |
| **Transformers** | OWLv2 model integration                |

---

# 🚀 Installation

## 📋 Requirements

Recommended:

* Python **3.10+**
* Git
* Webcam
* Internet connection
* At least **8 GB RAM**
* GPU recommended for faster AI inference

The project can run on CPU, although inference performance may be lower.

---

## 🐧 Linux / Ubuntu

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install git python3 python3-pip python3-venv
```

Check:

```bash
python3 --version
git --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/kr1on1x/MediaPipeHolistic.git
cd MediaPipeHolistic
```

---

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 6. Run AEGIS

```bash
python main.py
```

Make sure your webcam is connected and available.

---

# 🪟 Windows

### 1. Install

Install:

* Python 3.10+
* Git

During Python installation enable:

```text
Add Python to PATH
```

---

### 2. Clone

Open PowerShell:

```powershell
git clone https://github.com/kr1on1x/MediaPipeHolistic.git
cd MediaPipeHolistic
```

---

### 3. Create virtual environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

---

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Run

```powershell
python main.py
```

---

# 🤖 OWLv2 Setup

OWLv2 is downloaded automatically from Hugging Face when it is initialized for the first time.

The project uses:

```text
google/owlv2-base-patch16-ensemble
```

The first initialization may take some time because the model files need to be downloaded and cached locally.

A basic OWLv2 test is available in:

```text
test_owl.py
```

Run:

```bash
python test_owl.py
```

Expected output:

```text
Loading OWLv2...
OWLv2 ONLINE
```

---

# 🎮 Controls

| Key / Gesture | Function            |
| ------------- | ------------------- |
| `Q` / `ESC`   | Exit                |
| `F`           | Toggle fullscreen   |
| `SPACE`       | Capture screenshot  |
| `T`           | Start object search |
| `ENTER`       | Execute search      |
| `V SIGN`      | Switch mode         |
| `PINCH`       | Lock onto object    |
| `OPEN HAND`   | Release target      |
| `THUMBS UP`   | Confirm target      |

---

# 📸 Screenshots

Project screenshots can be found in the repository:

```text
aegis_1787861642.png
aegis_1787862743.png
```

More screenshots and demonstrations will be added as the project develops.

---

# 📁 Project Structure

```text
MediaPipeHolistic/
│
├── main.py
│   └── Main AEGIS application
│
├── test_owl.py
│   └── OWLv2 initialization test
│
├── yolo11n.pt
│   └── YOLO11n model weights
│
├── aegis_*.png
│   └── Captured screenshots
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│
├── .python-version
│
└── README.md
```

---

# 🔬 Development Goals

AEGIS is an experimental project and is continuously evolving.

Future development may include:

* [ ] More accurate gesture recognition
* [ ] Improved target tracking
* [ ] Object tracking between frames
* [ ] Better OWLv2 integration
* [ ] Voice commands
* [ ] More interaction modes
* [ ] GPU optimization
* [ ] Multi-camera support
* [ ] Improved UI/HUD
* [ ] Persistent target tracking
* [ ] Recording and replay system
* [ ] Modular AI pipeline
* [ ] Performance profiling
* [ ] Real-time event system

---

# 📊 Performance

AEGIS displays real-time FPS directly in the HUD.

Performance depends on:

* CPU
* GPU
* RAM
* Camera resolution
* YOLO inference resolution
* OWLv2 inference frequency
* Number of active AI models

The application uses asynchronous OWLv2 search so that open-vocabulary inference does not completely block the main camera loop.

---

# ⚠️ Notes

### Model files

The repository contains:

```text
yolo11n.pt
```

OWLv2 model files are downloaded automatically and cached by the Hugging Face Transformers ecosystem.

### Virtual environment

The `.venv` directory is intentionally excluded from Git.

Each machine should create its own virtual environment.

---

# 📜 License

This project is currently intended for **educational, experimental and research purposes**.

A dedicated open-source license may be added in the future.

---

# 👨‍💻 Author

**kr1on1x**

GitHub:

https://github.com/kr1on1x

---

<p align="center">
  <b>AEGIS</b>
  <br>
  <sub>Multimodal Vision Core</sub>
</p>
