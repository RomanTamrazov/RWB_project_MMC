# 🎥 RWB Final Progect MMC / Team Nam Ne DANO

Проект по **распознаванию действий, в реальном времени и видио** с использованием компьютерного зрения и позных моделей.  
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

camera / image / video

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

- **Python 3.9+**
- **OpenCV**
- **YOLOv8 (Ultralytics)** — детекция человека
- **MediaPipe Pose & Face Detection**
- **NumPy**
- **Telegram Bot API**

---

## 📥 Установка

### 1️⃣ Клонировать репозиторий
```bash
git clone 
cd 

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
## И возвращает видео/картинку с:
- bounding box человека
- действием
- намерением
- эмоцией
## Запуск:
```bash
python app/bot.py
```
### 👤 Авторы 
## Роман Тамразов и Динар Кугушев
## ML / Computer Vision
Проект разработан для исследовательских и образовательных целей.
