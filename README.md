# AI Security Surveillance System

An AI-powered security surveillance project built using Python, OpenCV, and YOLO. The system explores real-time video monitoring, person detection and tracking, weapon detection, restricted-area monitoring, event logging, and Telegram-based security alerts.

## Overview

The **AI Security Surveillance System** is a computer vision project designed to demonstrate how AI can assist with security monitoring through webcam video feeds.

The project is developed in multiple phases, starting with basic webcam access and progressing to object detection, person tracking, event monitoring, and an integrated surveillance dashboard.

## Features

- **Real-Time Video Monitoring:** Capture and process webcam video using OpenCV.
- **Person Detection:** Detect people in video frames using YOLO.
- **Person Counting and Tracking:** Track detected people and monitor movement.
- **Entry and Exit Monitoring:** Experiment with identifying entry and exit events.
- **Restricted-Area Monitoring:** Detect activity within configured restricted zones.
- **Weapon Detection:** Use a trained model to detect potential weapons.
- **Risk-Based Monitoring:** Combine detection results with risk-related logic.
- **Event Logging:** Record surveillance events in a CSV log file.
- **Surveillance Dashboard:** Integrate detection and monitoring features into a dashboard.
- **Telegram Alerts:** Send security notifications and snapshots through Telegram when configured.

*Note: Feature availability depends on the script being executed and its configuration.*

## Technology Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO
- **Deep Learning:** PyTorch (through the YOLO ecosystem)
- **Data Logging:** CSV
- **Notifications:** Telegram Bot API
- **Development Environment:** Visual Studio Code

## Project Structure

```text
AI-Security-Surveillance-System/
│
├── Phase1_openCV/
│   └── open_cv.py
│
├── Phase2_yolo/
│   ├── bus.jpg
│   └── yolo_test.py
│
├── Phase3_YOLO_Webcam/
│   └── yolo_webcam.py
│
├── Phase4_Person_detection/
│   ├── 1.only_person.py
│   ├── 2.person_counter.py
│   ├── 3.person_tracking.py
│   ├── 4.tracking_counter.py
│   ├── 5.person_entry_exit.py
│   ├── 6.stable_entry_exit.py
│   ├── 7.person_reidentification.py
│   ├── 8.restricted_area.py
│   ├── 9.event_logging.py
│   ├── 10.weapon_detection.py
│   ├── 11.integrated_detection.py
│   ├── 12.integrated_detection_v2.py
│   ├── 13.integrated_detection_with_risk.py
│   ├── 14.surveillance_dashboard.py
│   ├── 15.surveillance_dashboard_v2.py
│   ├── 16.surveillance_dashboard_v3.py
│   ├── 17.test_telegram.py
│   ├── 18.telegram_alert.py
│   ├── 20.dashboard_telegram_alert.py
│   └── telegram_alert.py
│
├── logs/
│   └── surveillance_log.csv
│
├── images.jpg
├── weapon_model.pt
├── yolo11n.pt
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python installed on your system.
- A webcam for live video monitoring.
- Git installed to clone the repository.
- A compatible computer capable of running the selected detection models.

### 1. Clone the Repository

```bash
git clone https://github.com/RajdipJana/AI-Security-Surveillance-System.git
cd AI-Security-Surveillance-System
```

### 2. Create a Virtual Environment

On Windows, run:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For scripts that use Telegram notifications, install `requests` if it is not already installed:

```powershell
python -m pip install requests
```

### 4. Run the Project

Start with the basic webcam demonstration:

```powershell
python Phase1_openCV/open_cv.py
```

Run the YOLO webcam demonstration:

```powershell
python Phase3_YOLO_Webcam/yolo_webcam.py
```

Explore the person detection and surveillance scripts in `Phase4_Person_detection/`. For example:

```powershell
python Phase4_Person_detection/14.surveillance_dashboard.py
```

Choose the appropriate script for the feature you want to test. Press `q` to exit webcam windows where that keyboard shortcut is implemented.

## Telegram Alert Configuration

Telegram notifications require your own bot credentials.

1. Create or configure a Telegram bot using [@BotFather](https://t.me/BotFather).
2. Obtain the bot token and your chat ID.
3. Create a local file named `telegram_config.py` in the project root.
4. Add your credentials to the local file:

```python
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

5. Ensure `telegram_config.py` is listed in `.gitignore`.
6. Configure the Telegram alert scripts to import these values.

**Security:** Never commit or publish your real bot token, chat ID, passwords, or other private credentials. Anyone cloning the repository must configure their own credentials to use Telegram alerts.

## Model Files

The repository includes the following model files:

- `yolo11n.pt` — YOLO model weights used for object detection.
- `weapon_model.pt` — model weights intended for weapon detection.

The exact model requirements depend on the script being run. Model predictions can be incorrect and should not be treated as definitive evidence of a security threat.

## Learning Objectives

This project provides practical experience with:

- Computer vision and image processing.
- Real-time object detection using YOLO.
- Person tracking and movement analysis.
- Event-based monitoring and logging.
- Integrating AI models into a surveillance workflow.
- Sending automated notifications through an API.
- Developing and organizing a multi-phase Python project.

## Limitations

- Detection accuracy depends on the model, video quality, lighting, and camera placement.
- Webcam availability and system performance may affect real-time processing.
- Weapon detection and risk classification require careful testing to reduce false positives and false negatives.
- This project is a learning and development prototype, not a certified security system.

## Future Improvements

- Improve detection accuracy and tracking stability.
- Enhance the surveillance dashboard and event history.
- Add configurable monitoring zones.
- Improve alert reliability and notification controls.
- Add more testing and documentation.

## Author

**Rajdip Jana**

B.Tech in Computer Science and Engineering (AI & ML)

GitHub: [@RajdipJana](https://github.com/RajdipJana)

---

If you find a bug or have a suggestion, feel free to open an issue in the repository.

**Disclaimer:** This project is intended for educational and research purposes. Use surveillance features responsibly and comply with applicable privacy laws and regulations.
