# 🎥 RWB Final Progect MMC / Team Nam Ne DANO

![Project Logo](kirill.gif)

Проект по **распознаванию действий, в реальном времени и видео** с использованием компьютерного зрения.  
Система работает **на CPU**.

---

## 🚀 Возможности проекта

В реальном времени система умеет:

### ✋ Жесты 
- статика
- машет рукой  
- шаг
- прыжок
- присед
---

## 🧠 Архитектура системы

Проект построен по **модульному принципу**:

camera / video

│

▼

PersonDetector (YOLOv8)

│

▼

PoseEstimator (MediaPipe Pose)

│

└── IntentPredictor (кинематика + логика)

│

▼

UI / Video / Telegram Bot


---

## 📦 Используемые технологии

- **Python 3.11**
- **OpenCV**
- **YOLOv8 (Ultralytics)** — детекция человека
- **MediaPipe Pose**
- **NumPy**
- **Telegram Bot API**

---

## 📥 Установка

### 1️⃣ Клонировать репозиторий
```bash
git clone git@github.com:RomanTamrazov/RWB_project_MMC.git
cd RWB_project_MMC

python -m venv venv

source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt

python app/main.py

```
---
### 🤖 Telegram-бот 
## Бот принимает:
- 🎥 видео
## И возвращает видео с:
- bounding box человека
- действием
- скелетом
## И вместе с видео возвращает 3D сцену со скелетом 
## Запуск:
```bash
python app/bot.py
```
### 👤 Авторы 
## Роман Тамразов и Динар Кугушев
## ML / Computer Vision
Проект разработан для исследовательских и образовательных целей.
