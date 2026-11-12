import gradio as gr
from PIL import Image

from pipeline_ocr import VietnameseHandwrittenOCR

# ==============================================================================
# 1. KHỞI TẠO ENGINE NHẬN DẠNG DÙNG CHUNG TỪ PIPELINE_OCR (LUÔN BẬT SỬA CHÍNH TẢ)
# ==============================================================================
print("\n" + "=" * 60)
print("⚡ [1/2] Đang nạp hệ thống Nhận dạng Chữ Viết Tay...")
ocr_engine = VietnameseHandwrittenOCR(use_corrector=True)
print("=" * 60 + "\n")


# ==============================================================================
# 2. QUY TRÌNH NHẬN DẠNG & TẢI ẢNH MỚI
# ==============================================================================
def process_ocr(input_image, mode):
    if input_image is None:
        return "Vui lòng chọn hoặc tải một ảnh chữ viết tay.", []

    is_multiline = mode != "1 Dòng / Từ đơn"
    return ocr_engine.predict_lines(
        input_image,
        is_multiline=is_multiline,
        apply_correction=True,
    )


def load_new_file(file):
    """Nạp ảnh trực tiếp khi bấm nút Chọn ảnh khác."""
    if file is None:
        return None
    file_path = file.name if hasattr(file, "name") else file
    return Image.open(file_path)


# ==============================================================================
# 3. GIAO DIỆN NỀN TRẮNG HIỆN ĐẠI (30% - 70%, TƯƠNG THÍCH GRADIO 6)
# ==============================================================================
custom_css = """
:root, .dark, body, .gradio-container {
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-label-background-fill: #f1f5f9 !important;
    --block-label-text-color: #334155 !important;
    --block-title-text-color: #0f172a !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --input-background-fill: #f8fafc !important;
    --input-border-color: #cbd5e1 !important;
    --border-color-primary: #e2e8f0 !important;
    --panel-background-fill: #ffffff !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-border-color: #cbd5e1 !important;
    --checkbox-label-background-fill: #ffffff !important;
    --checkbox-label-background-fill-hover: #f8fafc !important;
    --checkbox-label-background-fill-selected: #ffffff !important;
    --checkbox-label-background-fill-focus: #ffffff !important;
    --checkbox-label-text-color: #000000 !important;
    --checkbox-label-text-color-selected: #000000 !important;
    --checkbox-label-border-color: #cbd5e1 !important;
    --checkbox-label-border-color-selected: #2563eb !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-text-color: #334155 !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.app-title h2 {
    color: #0f172a !important;
    text-align: center !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
}

.white-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
}

.mode-radio fieldset,
.mode-radio .wrap {
    background-color: transparent !important;
}

.mode-radio label {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 500 !important;
}

.mode-radio label:hover {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    color: #000000 !important;
    border-color: #94a3b8 !important;
}

.mode-radio label.selected {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #2563eb !important;
    font-weight: 600 !important;
}

.mode-radio label span {
    color: #000000 !important;
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

light_theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#f8fafc",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8fafc",
    background_fill_secondary_dark="#f8fafc",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_background_fill="#f1f5f9",
    block_label_background_fill_dark="#f1f5f9",
    block_label_text_color="#334155",
    block_label_text_color_dark="#334155",
    block_title_text_color="#0f172a",
    block_title_text_color_dark="#0f172a",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#f8fafc",
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#e2e8f0",
    checkbox_background_color="#ffffff",
    checkbox_background_color_dark="#ffffff",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_dark="#ffffff",
    checkbox_label_background_fill_hover="#f8fafc",
    checkbox_label_background_fill_hover_dark="#f8fafc",
    checkbox_label_background_fill_selected="#ffffff",
    checkbox_label_background_fill_selected_dark="#ffffff",
    checkbox_label_text_color="#000000",
    checkbox_label_text_color_dark="#000000",
    checkbox_label_text_color_selected="#000000",
    checkbox_label_text_color_selected_dark="#000000",
    checkbox_label_border_color="#cbd5e1",
    checkbox_label_border_color_dark="#cbd5e1",
    checkbox_label_border_color_selected="#2563eb",
    checkbox_label_border_color_selected_dark="#2563eb",
)

with gr.Blocks(title="Nhận dạng Chữ Viết Tay") as demo:
    gr.Markdown(
        "## ✍️ Hệ thống Nhận dạng Chữ Viết Tay Tiếng Việt",
        elem_classes=["app-title"],
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
                elem_classes=["mode-radio"],
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

    btn_upload.upload(
        fn=load_new_file, inputs=[btn_upload], outputs=[input_image]
    )

    btn_submit.click(
        fn=process_ocr,
        inputs=[input_image, mode_radio],
        outputs=[output_text, gallery_output],
    )

# ==============================================================================
# 4. KHỞI CHẠY TỰ ĐỘNG & BẢO VỆ CỔNG MẠNG
# ==============================================================================
if __name__ == "__main__":
    port = 7860
    print("\n" + "=" * 65)
    print("🚀 GIAO DIỆN WEB ĐANG SẴN SÀNG!")
    print(f"👉 ĐƯỜNG LINK TRUY CẬP: http://127.0.0.1:{port}")
    print("🌐 Trình duyệt web sẽ tự động mở tab mới ngay bây giờ...")
    print("=" * 65 + "\n")

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=True,
            css=custom_css,
            theme=light_theme,
        )
    except OSError:
        alt_port = 7861
        print(f"\n⚠️ Cổng {port} đang bận! Tự động đổi sang cổng: {alt_port}")
        print(f"👉 ĐƯỜNG LINK MỚI: http://127.0.0.1:{alt_port}\n")
        demo.launch(
            server_name="127.0.0.1",
            server_port=alt_port,
            inbrowser=True,
            css=custom_css,
            theme=light_theme,
        )
