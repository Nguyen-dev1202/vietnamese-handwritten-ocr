import os
import cv2
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


def normalize_paper(cv_gray):
    """Làm phẳng ánh sáng nền giấy và giữ nét chữ sắc nét trên nền trắng."""
    h, w = cv_gray.shape
    if max(h, w) < 700:
        scale = min(2.5, 700.0 / max(h, w))
        cv_gray = cv2.resize(cv_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = cv_gray.shape

    max_dim = max(h, w)
    if max_dim > 500:
        scale = 500.0 / max_dim
        small = cv2.resize(
            cv_gray,
            (max(32, int(w * scale)), max(32, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = cv_gray

    sh, sw = small.shape
    k = max(15, int(min(sh, sw) * 0.06) | 1)
    k = min(k, 35)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (k, k), 0)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    norm = (cv_gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0
    norm = np.clip(norm, 0, 255)

    out = np.where(norm >= 226.0, 255.0, (norm / 226.0) * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def deskew_image(norm_gray):
    """Cân chỉnh độ nghiêng văn bản dựa trên cực đại hóa phương sai hình chiếu ngang."""
    h, w = norm_gray.shape
    scale = min(1.0, 700.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(
            norm_gray, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        small = norm_gray

    _, thresh = cv2.threshold(small, 205, 255, cv2.THRESH_BINARY_INV)
    sh, sw = thresh.shape
    center = (sw // 2, sh // 2)
    y0, y1 = int(sh * 0.08), int(sh * 0.92)
    x0, x1 = int(sw * 0.08), int(sw * 0.92)

    best_angle, max_var = 0.0, -1.0
    for angle in np.arange(-6.0, 6.5, 0.5):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(thresh, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        proj = np.sum(rot[y0:y1, x0:x1], axis=1)
        var = np.var(proj)
        if var > max_var:
            max_var = var
            best_angle = angle

    for angle in np.arange(best_angle - 0.4, best_angle + 0.45, 0.2):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(thresh, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        proj = np.sum(rot[y0:y1, x0:x1], axis=1)
        var = np.var(proj)
        if var > max_var:
            max_var = var
            best_angle = angle

    if abs(best_angle) < 0.25:
        return norm_gray
    M_best = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    return cv2.warpAffine(norm_gray, M_best, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def remove_ruled_and_margin_lines(norm_gray, is_multiline=True):
    """Loại bỏ dòng kẻ vở ngang và đường lề/chia cột dọc mà vẫn bảo toàn nét chữ giao cắt."""
    h, w = norm_gray.shape
    cleaned = norm_gray.copy()
    if min(h, w) < 40:
        return cleaned, []

    rule_r = 2 if h < 250 else (3 if h < 700 else 4)
    dark = norm_gray < 230

    # 1. Xử lý dòng kẻ ngang theo từng phân đoạn chiều rộng
    padded_y = np.pad(norm_gray, ((rule_r, rule_r), (0, 0)), constant_values=255)
    above_white = padded_y[:-2 * rule_r, :] > 200
    below_white = padded_y[2 * rule_r:, :] > 200
    thin_h_px = dark & above_white & below_white

    chunk_w = max(30, w // 3)
    for start_x in range(0, w, chunk_w):
        end_x = min(w, start_x + chunk_w)
        if (end_x - start_x) < max(20, w * 0.25):
            continue
        cov = thin_h_px[:, start_x:end_x].astype(np.float32).mean(axis=1)
        band_cov = ndimage.maximum_filter1d(cov, size=3)
        for y in np.where(band_cov > 0.32)[0]:
            y0, y1 = max(0, y - 1), min(h, y + 2)
            sub_thin = thin_h_px[y0:y1, start_x:end_x]
            cleaned[y0:y1, start_x:end_x][sub_thin] = 255

    # 2. Xử lý đường lề dọc và đường chia cột bảng (chỉ ở chế độ nhiều dòng)
    interior_dividers = []
    if is_multiline and h >= 90:
        padded_x = np.pad(norm_gray, ((0, 0), (rule_r, rule_r)), constant_values=255)
        left_white = padded_x[:, :-2 * rule_r] > 200
        right_white = padded_x[:, 2 * rule_r:] > 200
        thin_v_px = dark & left_white & right_white

        col_thin_cov = thin_v_px.astype(np.float32).mean(axis=0)
        col_dark_cov = (norm_gray < 215).astype(np.float32).mean(axis=0)
        v_cols_mask = (ndimage.maximum_filter1d(col_thin_cov, size=3) > 0.35) | (col_dark_cov > 0.55)

        v_idx = np.where(v_cols_mask)[0]
        if len(v_idx) > 0:
            groups = np.split(v_idx, np.where(np.diff(v_idx) > 3)[0] + 1)
            for grp in groups:
                cx = int(np.mean(grp))
                x0 = max(0, int(grp.min()) - 2)
                x1 = min(w, int(grp.max()) + 3)
                sub_thin_v = thin_v_px[:, x0:x1]
                cleaned[:, x0:x1][sub_thin_v] = 255
                if int(w * 0.15) < cx < int(w * 0.85) and (col_dark_cov[grp].max() > 0.45):
                    interior_dividers.append(cx)

    return cleaned, sorted(interior_dividers)


def tight_crop_and_pad(crop_gray):
    """Cắt sát vùng mực chữ thật (loại dòng trống/nhiễu) và đệm lề trắng chuẩn UIT-HWDB."""
    h, w = crop_gray.shape
    ink = (crop_gray < 185).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    valid = np.zeros_like(ink)
    total_area = 0
    min_comp_area = max(3, int((h * w) * 0.00004))

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        bx = stats[lbl, cv2.CC_STAT_LEFT]
        by = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        if area < min_comp_area:
            continue
        if bh <= max(2, int(h * 0.05)) and bw > max(20, int(w * 0.18)):
            continue
        # Loại bỏ vệt dòng kẻ ngang mảnh hơi nghiêng (mật độ mực theo chiều cao rất thấp)
        if bw > max(35, int(w * 0.18)) and (area / max(bw, 1)) <= 4.5 and bh <= max(14, int(h * 0.35)):
            continue
        if bw <= max(2, int(w * 0.015)) and bh > max(20, int(h * 0.65)):
            continue
        if (bx <= 1 or bx + bw >= w - 1) and bw <= 3 and bh < h * 0.45:
            continue
        if (by <= 1 or by + bh >= h - 1) and bh <= 3 and bw > w * 0.25:
            continue
        valid[labels == lbl] = 1
        total_area += area

    if total_area < max(25, int(h * 0.6)):
        return None

    # Loại bỏ nhiễu lẻ nằm tách biệt quá xa ở hai đầu trái/phải
    col_sums = valid.sum(axis=0)
    inked_cols = int((col_sums > 0).sum())
    if inked_cols == 0:
        return None

    ys, xs = np.where(valid > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    text_h = y2 - y1
    text_w = x2 - x1
    if text_h < 6 or text_w < 10:
        return None
    if text_w > w * 0.25 and text_h <= max(14, int(h * 0.35)) and (total_area / inked_cols) <= 4.2:
        return None

    gap_limit = max(60, int(text_h * 2.2))
    ink_runs = np.where(col_sums > 0)[0]
    gaps = np.where(np.diff(ink_runs) > gap_limit)[0]
    if len(gaps) > 0:
        # Giữ lại dải cụm mực chính, bỏ cụm nhiễu < 2.5% tổng diện tích ở ngoài rìa xa
        split_pts = [0] + [int(g + 1) for g in gaps] + [len(ink_runs)]
        clusters = []
        for idx in range(len(split_pts) - 1):
            c_cols = ink_runs[split_pts[idx] : split_pts[idx + 1]]
            c_area = int(col_sums[c_cols].sum())
            clusters.append((int(c_cols[0]), int(c_cols[-1]) + 1, c_area))
        kept = [c for c in clusters if c[2] >= max(35, total_area * 0.025)]
        if kept:
            x1 = min(c[0] for c in kept)
            x2 = max(c[1] for c in kept)
            ys_k = np.where(valid[:, x1:x2] > 0)[0]
            if len(ys_k) > 0:
                y1, y2 = int(ys_k.min()), int(ys_k.max()) + 1
                text_h = y2 - y1
                text_w = x2 - x1

    pad_y = max(3, int(text_h * 0.08))
    pad_x = max(5, int(text_h * 0.06))

    y1 = max(0, y1 - 1)
    y2 = min(h, y2 + 1)
    x1 = max(0, x1 - 2)
    x2 = min(w, x2 + 2)
    sub = crop_gray[y1:y2, x1:x2]
    return cv2.copyMakeBorder(sub, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)


def _segment_region_lines(region_gray):
    """Phân đoạn các dòng chữ trong một vùng ảnh đã làm sạch."""
    h, w = region_gray.shape
    ink_bin = (region_gray < 185).astype(np.uint8)
    proj = np.sum(ink_bin[:, int(w * 0.02):int(w * 0.98)] > 0, axis=1).astype(float)
    smoothed = ndimage.gaussian_filter1d(proj, sigma=max(2.5, h / 80.0))
    max_val = np.max(smoothed)
    if max_val < 1.5:
        return []

    min_dist = max(12, int(h / 18))
    peaks, _ = find_peaks(smoothed, distance=min_dist, prominence=max_val * 0.05)

    if len(peaks) <= 1:
        cropped = tight_crop_and_pad(region_gray)
        return [cropped] if cropped is not None else []

    ink_idx = np.where(smoothed > max_val * 0.02)[0]
    g_top = max(0, int(ink_idx[0]) - 6) if len(ink_idx) else 0
    g_bot = min(h, int(ink_idx[-1]) + 6) if len(ink_idx) else h

    valleys = [p1 + int(np.argmin(smoothed[p1:p2])) for p1, p2 in zip(peaks[:-1], peaks[1:])]

    crops = []
    for i in range(len(peaks)):
        top_y = g_top if i == 0 else max(0, valleys[i - 1] - 2)
        bot_y = g_bot if i == len(peaks) - 1 else min(h, valleys[i] + 2)
        if (bot_y - top_y) >= 8:
            raw_crop = region_gray[top_y:bot_y, :]
            cropped = tight_crop_and_pad(raw_crop)
            if cropped is not None:
                crops.append(cropped)
    return crops


def preprocess_and_segment(image, is_multiline=True):
    """Tiền xử lý toàn diện (làm phẳng nền, xoay thẳng, xóa dòng kẻ) và tách dòng chữ."""
    if isinstance(image, Image.Image):
        cv_gray = np.array(image.convert("L"))
    elif image.ndim == 3:
        cv_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        cv_gray = image.copy()

    norm = normalize_paper(cv_gray)
    deskewed = deskew_image(norm)
    cleaned, v_dividers = remove_ruled_and_margin_lines(deskewed, is_multiline=is_multiline)

    if not is_multiline:
        cropped = tight_crop_and_pad(cleaned)
        return [cropped] if cropped is not None else [cv2.copyMakeBorder(cleaned, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)]

    if v_dividers:
        bounds = [0] + v_dividers + [cleaned.shape[1]]
        all_crops = []
        for idx in range(len(bounds) - 1):
            x0 = bounds[idx] + (4 if idx > 0 else 0)
            x1 = bounds[idx + 1] - (4 if idx < len(bounds) - 2 else 0)
            if (x1 - x0) >= 25:
                all_crops.extend(_segment_region_lines(cleaned[:, x0:x1]))
        if all_crops:
            return all_crops

    crops = _segment_region_lines(cleaned)
    if crops:
        return crops
    return [cv2.copyMakeBorder(cleaned, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)]


class VietnameseHandwrittenOCR:
    def __init__(self, weights_path=None, config_path=None, use_corrector=True):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = weights_path or os.path.join(BASE_DIR, "weights", "vietocr_hw_best.pth")
        config_path = config_path or os.path.join(BASE_DIR, "weights", "config_line.yml")
        if not os.path.exists(config_path):
            config_path = os.path.join(BASE_DIR, "weights", "config.yml")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Đang nạp mô hình VietOCR...")
        config = Cfg.load_config_from_file(config_path)
        config["weights"] = weights_path
        config["device"] = self.device
        self.detector = Predictor(config)
        print(f"-> Nạp mô hình thành công trên thiết bị: {self.device}!")

        self.corrector_tok = None
        self.corrector_model = None
        self._corrector_failed = False
        if use_corrector:
            self.load_corrector()

    def load_corrector(self):
        """Nạp mô hình sửa lỗi chính tả tiếng Việt theo nhu cầu."""
        if self.corrector_model is not None or self._corrector_failed:
            return self.corrector_model is not None
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            print("Đang nạp mô hình sửa lỗi chính tả tiếng Việt...")
            model_name = "bmd1905/vietnamese-correction-v2"
            self.corrector_tok = AutoTokenizer.from_pretrained(model_name)
            self.corrector_model = (
                AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
            )
            print("-> Hệ thống sửa lỗi chính tả đã sẵn sàng!")
            return True
        except Exception as exc:
            print(f"⚠️ Không thể nạp mô hình sửa lỗi chính tả ({exc}). Tiếp tục chạy OCR thuần.")
            self._corrector_failed = True
            return False

    def deskew(self, cv_img):
        return deskew_image(normalize_paper(cv_img))

    def segment_lines(self, pil_img):
        mats = preprocess_and_segment(pil_img, is_multiline=True)
        return [Image.fromarray(m) for m in mats]

    def correct_spelling(self, text):
        if not text.strip():
            return text
        if self.corrector_model is None and not self.load_corrector():
            return text
        try:
            inputs = self.corrector_tok(
                text, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self.device)
            with torch.no_grad():
                outputs = self.corrector_model.generate(**inputs, max_length=128, num_beams=3)
            return self.corrector_tok.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return text

    def predict_lines(self, image_or_path, is_multiline=None, apply_correction=False):
        """Nhận dạng trả về văn bản đầy đủ và danh sách (PIL.Image, text) từng dòng."""
        img = Image.open(image_or_path) if isinstance(image_or_path, (str, os.PathLike)) else image_or_path
        if is_multiline is None:
            if isinstance(img, Image.Image):
                w, h = img.size
            else:
                h, w = img.shape[:2]
            is_multiline = (h / max(w, 1)) >= 0.25

        line_mats = preprocess_and_segment(img, is_multiline=is_multiline)
        texts = []
        gallery_items = []
        for mat in line_mats:
            pil_line = Image.fromarray(mat)
            raw = self.detector.predict(pil_line).strip()
            if not raw:
                continue
            final_text = self.correct_spelling(raw) if apply_correction else raw
            texts.append(final_text)
            gallery_items.append((pil_line, f"Dòng {len(texts)}: {final_text}"))

        return "\n".join(texts), gallery_items

    def predict(self, img_path):
        full_text, _ = self.predict_lines(
            img_path, is_multiline=None, apply_correction=(self.corrector_model is not None)
        )
        return full_text


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(BASE_DIR, "UIT_HWDB_paragraph", "test_data", "251", "4.jpg")

    ocr = VietnameseHandwrittenOCR(use_corrector=False)
    print("\n--- KẾT QUẢ NHẬN DẠNG ---")
    output = ocr.predict(sample_img)
    print(output)
