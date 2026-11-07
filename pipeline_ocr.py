import os
import sys
import cv2
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

class VietnameseHandwrittenOCR:
    def __init__(self, weights_path=None, config_path=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = weights_path or os.path.join(BASE_DIR, 'weights', 'vietocr_hw_best.pth')
        config_path = config_path or os.path.join(BASE_DIR, 'weights', 'config_line.yml')

        print("Đang nạp mô hình VietOCR...")
        config = Cfg.load_config_from_file(config_path)
        config['weights'] = weights_path
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.detector = Predictor(config)

        print("Đang nạp mô hình sửa lỗi chính tả tiếng Việt...")
        model_name = "bmd1905/vietnamese-correction-v2"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.corrector_tok = AutoTokenizer.from_pretrained(model_name)
        self.corrector_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        print("-> Hệ thống OCR đã sẵn sàng!")

    def deskew(self, cv_img):
        _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = thresh.shape
        center = (w // 2, h // 2)
        best_angle, max_var = 0, 0
        for angle in np.arange(-6.0, 6.5, 0.5):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            proj = np.sum(rot[:, int(w*0.05):int(w*0.95)], axis=1)
            var = np.var(proj)
            if var > max_var:
                max_var = var
                best_angle = angle
        M_best = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        return cv2.warpAffine(cv_img, M_best, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    def segment_lines(self, pil_img):
        cv_gray = np.array(pil_img.convert('L'))
        deskewed = self.deskew(cv_gray)
        h, w = deskewed.shape

        _, thresh = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(thresh[:, int(w * 0.03):int(w * 0.97)], axis=1).astype(float)
        smoothed = ndimage.gaussian_filter1d(proj, sigma=5)

        peaks, _ = find_peaks(smoothed, distance=max(20, int(h / 15)), prominence=np.max(smoothed) * 0.04)
        if len(peaks) <= 1:
            return [pil_img]

        cut_y = []
        top_limit = max(0, peaks[0] - 50)
        cut_y.append(top_limit + np.argmin(smoothed[top_limit:peaks[0]]))

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            cut_y.append(p1 + np.argmin(smoothed[p1:p2]))

        bot_limit = min(h, peaks[-1] + 65)
        cut_y.append(peaks[-1] + np.argmin(smoothed[peaks[-1]:bot_limit]))

        line_imgs = []
        for i in range(len(cut_y) - 1):
            y1 = max(0, cut_y[i] - 10)
            y2 = min(h, cut_y[i + 1] + 14)
            if (y2 - y1) >= 15:
                crop = deskewed[y1:y2, 0:w]
                line_imgs.append(Image.fromarray(crop))
        return line_imgs

    def correct_spelling(self, text):
        if not text.strip():
            return text
        try:
            inputs = self.corrector_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.corrector_model.generate(**inputs, max_length=128, num_beams=3)
            return self.corrector_tok.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return text

    def predict(self, img_path):
        img = Image.open(img_path)
        w, h = img.size
        # Nếu là 1 từ hoặc 1 dòng đơn lẻ (ảnh dẹt)
        if (h / w) < 0.25:
            raw = self.detector.predict(img).strip()
            return self.correct_spelling(raw)
        
        # Nếu là đoạn văn bản nhiều dòng
        line_crops = self.segment_lines(img)
        res = []
        for line in line_crops:
            raw = self.detector.predict(line).strip()
            fixed = self.correct_spelling(raw)
            res.append(fixed)
        return "\n".join(res)

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(BASE_DIR, "UIT_HWDB_paragraph", "test_data", "251", "4.jpg")
    
    ocr = VietnameseHandwrittenOCR()
    print("\n--- KẾT QUẢ NHẬN DẠNG ---")
    output = ocr.predict(sample_img)
    print(output)