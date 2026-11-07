# ✍️ HỆ THỐNG NHẬN DẠNG CHỮ VIẾT TAY TIẾNG VIỆT (VIETNAMESE HTR)
> **Đồ án Capstone - Samsung Innovation Campus (SIC) AI Course 2025 - 2026 | Team 3**  
> **Thành viên:** Lưu Thanh Nguyên (Trưởng nhóm), Đặng Hoàng Anh

---

## 📌 Giới thiệu dự án
Dự án xây dựng hệ thống Nhận dạng chữ viết tay tiếng Việt (**Handwritten Text Recognition - HTR / OCR**) hoàn chỉnh từ đầu đến cuối (End-to-End) có khả năng xử lý linh hoạt từ **Từ đơn lẻ (Word)**, **Dòng chữ (Line)** cho đến **Đoạn văn bản hoàn chỉnh (Paragraph)**.

### ✨ Các tính năng nổi bật:
* **Kiến trúc học sâu VGG-Transformer:** Sử dụng mạng tích chập VGG19-BN trích xuất đặc trưng không gian kết hợp mạng Transformer Encoder-Decoder (VietOCR) với cơ chế Self-Attention đa đầu.
* **Tự động cân chỉnh độ nghiêng (Auto-Deskew):** Tối ưu hóa phương sai hình chiếu ngang để xoay thẳng văn bản trong dải góc từ -6° đến +6.5°.
* **Phân đoạn dòng thích ứng (Adaptive Line Segmentation):** Chiếu đồ thị mật độ pixel, lọc mịn Gaussian 1D thích ứng và định vị đỉnh/thung lũng thông minh.
* **Vùng đệm lề an toàn (Safe White Padding):** Tự động đệm viền trắng quanh từng dòng chữ trước khi nhận dạng, bảo toàn trọn vẹn 100% các dấu thanh cao (sắc, hỏi, ngã, mũ) và chân chữ thấp (g, y, p) tiếng Việt.
* **Hậu xử lý sửa lỗi chính tả ngữ cảnh:** Tích hợp mô hình ngôn ngữ Seq2Seq `bmd1905/vietnamese-correction-v2` giúp khôi phục ngữ nghĩa, nâng độ chính xác toàn văn bản lên trên 92% - 95%.
* **Giao diện Web Studio hiện đại (Gradio):** Thiết kế Clean White Dashboard chuẩn mực (30% Panel điều khiển - 70% Khung kết quả), hỗ trợ kiểm chứng trực quan ảnh từng dòng đã bóc tách.

---

## 📂 Cấu trúc thư mục dự án
```text
huan_luyen5/
├── weights/
│   ├── config_line.yml           # Cấu hình huấn luyện mô hình dòng
│   ├── config_full.yml           # Cấu hình tinh chỉnh mô hình từ
│   └── vietocr_hw_best.pth       # Trọng số tốt nhất (đặt tại đây)
├── app.py                        # Giao diện Web Studio Gradio
├── pipeline_ocr.py               # Module nhận dạng & hậu xử lý chính tả
├── train_line.py                 # Mã nguồn huấn luyện cấp độ dòng
├── train_word.py                 # Mã nguồn tinh chỉnh cấp độ từ
├── run_app.bat                   # File khởi chạy nhanh 1-click cho Windows
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt môi trường (Khuyên dùng Miniconda / Anaconda)
Khuyến nghị sử dụng **Python 3.10**:
```bash
# Tạo môi trường mới với Python 3.10
conda create -n ocr_sic python=3.10 -y
conda activate ocr_sic

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

### 2. Trọng số mô hình (Weights)
Do giới hạn dung lượng lưu trữ của GitHub (<100MB/file), file trọng số huấn luyện `vietocr_hw_best.pth` (~151MB) cần được đặt trực tiếp vào thư mục `weights/`:
```text
weights/vietocr_hw_best.pth
```

### 3. Khởi chạy Ứng dụng Web
* **Cách 1 (Nhanh nhất trên Windows):** Nhấp đúp chuột vào file **`run_app.bat`**.
* **Cách 2 (Dùng lệnh terminal):**
```bash
python app.py
```
Ứng dụng sẽ tự động khởi động tại địa chỉ: **`http://127.0.0.1:7860`**

---

## 📊 Kết quả Thực nghiệm
* **Độ chính xác ký tự (Character Accuracy):** ~87.2% trên tập dòng chữ và >94% trên tập từ đơn.
* **Độ chính xác chuỗi (Sequence Accuracy):** ~87.69%.
* **Độ chính xác sau sửa lỗi ngữ cảnh:** Đạt 92% - 95% trên các đoạn văn bản viết tay phức tạp.
