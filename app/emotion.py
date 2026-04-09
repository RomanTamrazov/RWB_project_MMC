import cv2
import numpy as np
from deepface import DeepFace                                                                

                                                                       
EMOTION_MAP = {
    "angry": "злой",
    "disgust": "отвращение",
    "fear": "испуган",
    "happy": "счастлив",
    "sad": "грустный",
    "surprise": "удивлён",
    "neutral": "нейтральная"
}

class EmotionRecognizer:
    def __init__(self):
                                                                                             
                                                                                     
        self.use_deepface = True                                                   
        try:
            import onnxruntime as ort
            import scipy.special
            self.session = ort.InferenceSession(
                "models/emotion/emotion-ferplus.onnx",
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.softmax = scipy.special.softmax            
            self.emotions_ru = [                                              
                "нейтральная", "счастлив", "удивлён", "грустный",
                "злой", "испуган", "отвращение", "презрение"
            ]
        except Exception as e:
            print(f"Предупреждение: ONNX не загружен ({e}). Используем только DeepFace.")
            self.use_deepface = True

    def predict(self, face_img):
        """
        Предсказывает эмоцию на изображении лица.
        Использует DeepFace по умолчанию (лучшая точность, меньше bias к neutral).
        Возвращает: (эмоция_ру, вероятность [0-1])
        """
        if self.use_deepface:
            try:
                                                                                   
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',                  
                    silent=True                    
                )
                emo_en = result[0]['dominant_emotion']
                emo_ru = EMOTION_MAP.get(emo_en, emo_en)                        
                emo_prob = result[0]['emotion'][emo_en] / 100.0         
                                                                    
                if emo_prob < 0.6:
                    return "неуверенно", emo_prob
                return emo_ru, emo_prob
            except Exception as e:
                print(f"Ошибка DeepFace: {e}. Fallback на ONNX если доступен.")
                if not hasattr(self, 'session'):
                    return "ошибка", 0.0

                                                                     
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray = gray.astype("float32") / 255.0
            gray = gray[np.newaxis, np.newaxis, :, :]
            outputs = self.session.run(None, {self.input_name: gray})
            logits = outputs[0][0]
            probs = self.softmax(logits)
            idx = int(np.argmax(probs))
            emo_ru = self.emotions_ru[idx]
            emo_prob = float(probs[idx])
            if emo_prob < 0.6:
                return "неуверенно", emo_prob
            return emo_ru, emo_prob
        except Exception as e:
            print(f"Ошибка ONNX: {e}")
            return "ошибка", 0.0