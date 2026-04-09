from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

FONT_PATH = "app/fonts/DejaVuSans.ttf"

def draw_text(frame, text, pos, size=28):
    try:
        font = ImageFont.truetype(FONT_PATH, size)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox(pos, text, font=font)
        
                             
        padding = 2                 
        draw.rectangle([bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], fill=(0,0,0))
        draw.text(pos, text, font=font, fill=(255,255,255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"Ошибка в draw_text (PIL): {e}. Используем fallback OpenCV.")
        scale = size / 28
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        padding = 2                      
        cv2.rectangle(frame, (pos[0]-padding, pos[1]-text_size[1]-padding), 
                      (pos[0]+text_size[0]+padding, pos[1]+padding), (0,0,0), -1)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        return frame
