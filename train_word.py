import os
import sys
import json
import shutil
import random
import torch

# ==============================================================================
# 1. VÁ LỖI TƯƠNG THÍCH NUMPY 2.0 TOÀN DIỆN (IMGAUG + VIETOCR)
# ==============================================================================
import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }
if not hasattr(np, "bool"):
    np.bool = bool

# Chuyển đổi an toàn np.fromstring sang np.frombuffer khi đọc byte ảnh
_orig_fromstring = np.fromstring
def _safe_fromstring(string, dtype=float, count=-1, sep=""):
    if sep == "" or sep is None:
        return np.frombuffer(string, dtype=dtype, count=count)
    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
np.fromstring = _safe_fromstring
# ==============================================================================

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.model.vocab import Vocab

# Vá an toàn hàm encode từ điển trực tiếp trong bộ nhớ (Chống văng KeyError)
_orig_vocab_encode = Vocab.encode
def _safe_vocab_encode(self, chars):
    return [self.go] + [self.c2i[c] for c in chars if c in self.c2i] + [self.eos]
Vocab.encode = _safe_vocab_encode


def parse_and_clean_word_data(base_word_dir, split_name, output_txt):
    split_dir = os.path.join(base_word_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"Cảnh báo: Không tìm thấy thư mục {split_dir}")
        return []

    subfolders = [os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    cleaned_samples = []

    print(f"Đang quét tập {split_name} ({len(subfolders)} thư mục con)...")
    for sub in subfolders:
        label_file = None
        for fname in ['label', 'label.json', 'label.txt']:
            cand = os.path.join(sub, fname)
            if os.path.isfile(cand):
                label_file = cand
                break
        if not label_file:
            continue

        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for img_name, label in data.items():
                label = str(label).strip()
                # Lọc bỏ các mẫu chứa ký tự 'f'/'F' không có trong từ điển tiếng Việt chuẩn
                if label and ('f' not in label.lower()):
                    img_path = os.path.join(sub, img_name)
                    if os.path.exists(img_path):
                        rel_path = os.path.relpath(img_path, base_word_dir)
                        cleaned_samples.append(f"{rel_path}\t{label}")
        except Exception:
            continue

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_samples))
    print(f"-> Đã ghi nhận {len(cleaned_samples)} mẫu sạch vào: {output_txt}")
    return cleaned_samples


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    word_dir = os.path.join(BASE_DIR, "UIT_HWDB_word")
    weights_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Dọn dẹp cache train_data / valid_data nếu bị ngắt dở dang
    for temp_folder in ['train_data', 'valid_data']:
        p = os.path.join(BASE_DIR, temp_folder)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    # 2. Chuẩn bị file nhãn annotation sạch
    train_word_ann = os.path.join(BASE_DIR, "train_word_clean.txt")
    val_word_ann = os.path.join(BASE_DIR, "val_word_clean.txt")

    if not os.path.exists(train_word_ann):
        parse_and_clean_word_data(word_dir, "train_data", train_word_ann)
    else:
        print(f"-> Đã có sẵn file nhãn: {train_word_ann}")

    if not os.path.exists(val_word_ann):
        parse_and_clean_word_data(word_dir, "test_data", val_word_ann)
    else:
        print(f"-> Đã có sẵn file nhãn: {val_word_ann}")

    # 3. Cấu hình mô hình
    config_line_path = os.path.join(weights_dir, 'config_line.yml')
    if os.path.exists(config_line_path):
        config = Cfg.load_config_from_file(config_line_path)
    else:
        config = Cfg.load_config_from_name('vgg_transformer')

    config['dataset']['data_root'] = word_dir
    config['dataset']['train_annotation'] = train_word_ann
    config['dataset']['valid_annotation'] = val_word_ann

    config['trainer']['export'] = os.path.join(weights_dir, 'vietocr_hw_full.pth')
    config['trainer']['checkpoint'] = os.path.join(weights_dir, 'checkpoint_word.pth')
    config['trainer']['log'] = os.path.join(weights_dir, 'train_word.log')

    # Siêu tham số tối ưu chạy trên máy tính
    config['trainer']['batch_size'] = 16
    config['trainer']['print_every'] = 100
    config['trainer']['valid_every'] = 500
    config['trainer']['iters'] = 6000
    config['trainer']['lr'] = 5e-5
    config['dataloader']['num_workers'] = 0  # Chạy an toàn trên Windows
    config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config.save(os.path.join(weights_dir, 'config_full.yml'))

    # 4. Kế thừa từ mô hình Line tốt nhất (nếu có)
    line_weights = os.path.join(weights_dir, 'vietocr_hw_best.pth')
    if os.path.exists(line_weights):
        print(f"=== KẾ THỪA TRỌNG SỐ TỪ: {line_weights} ===")
        trainer = Trainer(config=config, pretrained=False)
        trainer.load_weights(line_weights)
    else:
        print("=== KHỞI TẠO MÔ HÌNH PRE-TRAINED GỐC ===")
        trainer = Trainer(config=config, pretrained=True)

    print(f"=== BẮT ĐẦU HUẤN LUYỆN WORD MODEL (Device: {config['device']}) ===")
    trainer.train()


if __name__ == '__main__':
    main()