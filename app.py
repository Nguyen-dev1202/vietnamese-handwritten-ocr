import os
import sys
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# ==============================================================================
# 1. KHỞI TẠO MÔ HÌNH NHẬN DẠNG VIETOCR
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "weights", "vietocr_hw_best.pth")
config_path = os.path.join(BASE_DIR, "weights", "config_line.yml")
if not os.path.exists(config_path):
  config_path = os.path.join(BASE_DIR, "weights", "config.yml")

print("\n" + "=" * 60)
print("⚡ [1/2] Đang nạp mô hình VietOCR vào bộ nhớ...")
config = Cfg.load_config_from_file(config_path)
config["weights"] = weights_path
config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = Predictor(config)
print(f"-> Nạp mô hình thành công trên thiết bị: {config['device']}!")
print("=" * 60 + "\n")


# ==============================================================================
# 2. XỬ LÝ ẢNH & TÁCH DÒNG BẢO TOÀN NÉT
# ==============================================================================
def deskew_image(cv_img):
  """Cân chỉnh độ nghiêng văn bản."""
  _, thresh = cv2.threshold(
      cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )
  h, w = thresh.shape
  center = (w // 2, h // 2)

  best_angle, max_var = 0, 0
  for angle in np.arange(-4.0, 4.5, 0.5):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    proj = np.sum(rot[:, int(w * 0.05) : int(w * 0.95)], axis=1)
    var = np.var(proj)
    if var > max_var:
      max_var = var
      best_angle = angle

  if abs(best_angle) < 0.5:
    return cv_img
  M_best = cv2.getRotationMatrix2D(center, best_angle, 1.0)
  return cv2.warpAffine(
      cv_img, M_best, (w, h), flags=cv2.INTER_CUBIC, borderValue=255
  )


def add_white_padding(cv_img, pad_x=28, pad_y=14):
  """Bổ sung lề trắng đệm để không bị mất nét đầu dòng hoặc dấu thanh."""
  return cv2.copyMakeBorder(
      cv_img,
      pad_y,
      pad_y,
      pad_x,
      pad_x,
      cv2.BORDER_CONSTANT,
      value=[255, 255, 255],
  )


def segment_multiple_lines(pil_img):
  """Tách dòng chính xác cho ảnh 2 dòng trở lên."""
  cv_gray = np.array(pil_img.convert("L"))
  deskewed = deskew_image(cv_gray)
  h, w = deskewed.shape

  _, thresh = cv2.threshold(
      deskewed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )
  proj = np.sum(thresh[:, int(w * 0.03) : int(w * 0.97)], axis=1).astype(float)
  smoothed = ndimage.gaussian_filter1d(proj, sigma=max(4, int(h / 60)))

  max_val = np.max(smoothed)
  if max_val == 0:
    return [add_white_padding(deskewed)]

  min_dist = max(20, int(h / 15))
  peaks, _ = find_peaks(smoothed, distance=min_dist, prominence=max_val * 0.06)

  if len(peaks) <= 1:
    return [add_white_padding(deskewed)]

  ink_indices = np.where(smoothed > max_val * 0.02)[0]
  global_top = max(0, ink_indices[0] - 12) if len(ink_indices) > 0 else 0
  global_bot = min(h, ink_indices[-1] + 12) if len(ink_indices) > 0 else h

  valleys = []
  for i in range(len(peaks) - 1):
    p1, p2 = peaks[i], peaks[i + 1]
    valleys.append(p1 + np.argmin(smoothed[p1:p2]))

  line_crops = []
  for i in range(len(peaks)):
    if i == 0:
      top_y = global_top
      bot_y = min(h, valleys[0] + 10)
    elif i == len(peaks) - 1:
      top_y = max(0, valleys[-1] - 10)
      bot_y = global_bot
    else:
      top_y = max(0, valleys[i - 1] - 8)
      bot_y = min(h, valleys[i] + 8)

    if (bot_y - top_y) >= 16:
      crop = deskewed[top_y:bot_y, 0:w]
      line_crops.append(add_white_padding(crop, pad_x=28, pad_y=12))

  return line_crops if line_crops else [add_white_padding(deskewed)]


# ==============================================================================
# 3. QUY TRÌNH NHẬN DẠNG & TẢI ẢNH MỚI
# ==============================================================================
def process_ocr(input_image, mode):
  if input_image is None:
    return "Vui lòng chọn hoặc tải một ảnh chữ viết tay.", []

  if isinstance(input_image, Image.Image):
    cv_img = np.array(input_image.convert("L"))
  else:
    cv_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

  if mode == "1 Dòng / Từ đơn":
    line_mats = [add_white_padding(cv_img, pad_x=28, pad_y=14)]
  else:
    line_mats = segment_multiple_lines(Image.fromarray(cv_img))

  recognized_texts = []
  gallery_items = []

  for idx, mat in enumerate(line_mats):
    pil_line = Image.fromarray(mat)
    text = detector.predict(pil_line).strip()
    if text:
      recognized_texts.append(text)
      gallery_items.append((pil_line, f"Dòng {idx + 1}: {text}"))

  return "\n".join(recognized_texts), gallery_items


def load_new_file(file):
  """Nạp ảnh trực tiếp khi bấm nút Chọn ảnh khác."""
  if file is None:
    return None
  file_path = file.name if hasattr(file, "name") else file
  return Image.open(file_path)


# ==============================================================================
# 4. GIAO DIỆN NỀN TRẮNG HIỆN ĐẠI (30% - 70%)
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.white-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
}

.btn-primary {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    background: #1d4ed8 !important;
}

.btn-secondary {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
}
.btn-secondary:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
}

.result-box textarea {
    font-size: 1.25rem !important;
    font-weight: 500 !important;
    line-height: 1.8 !important;
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

.custom-gallery img {
    object-fit: contain !important;
    background-color: #f1f5f9 !important;
    border-radius: 6px !important;
}
"""

with gr.Blocks(
    title="Nhận dạng Chữ Viết Tay",
    css=custom_css,
    theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"),
) as demo:
  gr.Markdown(
      "## ✍️ Hệ thống Nhận dạng Chữ Viết Tay Tiếng Việt",
      elem_classes=["text-center"],
  )

  with gr.Row():
    # CỘT TRÁI: 30%
    with gr.Column(scale=3, elem_classes=["white-card"]):
      input_image = gr.Image(
          type="pil", label="Ảnh chữ viết tay", show_label=False
      )

      btn_upload = gr.UploadButton(
          "📂 Chọn ảnh khác / Tải ảnh mới",
          file_types=["image"],
          variant="secondary",
          elem_classes=["btn-secondary"],
      )

      mode_radio = gr.Radio(
          choices=["Nhiều dòng (2 dòng trở lên)", "1 Dòng / Từ đơn"],
          value="Nhiều dòng (2 dòng trở lên)",
          label="Chế độ quét",
      )

      btn_submit = gr.Button(
          "🚀 BẮT ĐẦU NHẬN DẠNG",
          variant="primary",
          elem_classes=["btn-primary"],
      )

    # CỘT PHẢI: 70%
    with gr.Column(scale=7, elem_classes=["white-card"]):
      output_text = gr.Textbox(
          label="Kết quả nhận dạng",
          lines=8,
          placeholder="Nội dung chữ viết tay sẽ hiển thị đầy đủ tại đây...",
          elem_classes=["result-box"],
      )

      with gr.Accordion(
          "🔍 Kiểm chứng các dòng chữ được AI phân đoạn", open=True
      ):
        gallery_output = gr.Gallery(
            label="Ảnh từng dòng đã cắt",
            columns=1,
            height="auto",
            elem_classes=["custom-gallery"],
        )

  # Sự kiện tải ảnh mới
  btn_upload.upload(
      fn=load_new_file, inputs=[btn_upload], outputs=[input_image]
  )

  # Sự kiện nhận dạng
  btn_submit.click(
      fn=process_ocr,
      inputs=[input_image, mode_radio],
      outputs=[output_text, gallery_output],
  )

# ==============================================================================
# 5. KHỞI CHẠY TỰ ĐỘNG & BẢO VỆ CỔNG MẠNG
# ==============================================================================
if __name__ == "__main__":
  port = 7860
  print("\n" + "=" * 65)
  print("🚀 GIAO DIỆN WEB ĐANG SẴN SÀNG!")
  print(f"👉 ĐƯỜNG LINK TRUY CẬP: http://127.0.0.1:{port}")
  print("🌐 Trình duyệt web sẽ tự động mở tab mới ngay bây giờ...")
  print("=" * 65 + "\n")

  try:
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
  except OSError:
    alt_port = 7861
    print(f"\n⚠️ Cổng {port} đang bận! Tự động đổi sang cổng: {alt_port}")
    print(f"👉 ĐƯỜNG LINK MỚI: http://127.0.0.1:{alt_port}\n")
    demo.launch(server_name="127.0.0.1", server_port=alt_port, inbrowser=True)