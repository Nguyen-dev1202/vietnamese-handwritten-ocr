import os
import sys
import shutil
import random
import torch

# ==============================================================================
# VÁ LỖI TƯƠNG THÍCH TOÀN DIỆN CHO NUMPY 2.0 (IMGAUG + VIETOCR CREATE_DATASET)
# ==============================================================================
import numpy as np

# 1. Vá sctypes và bool cho imgaug
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

# 2. Vá np.fromstring chế độ binary chuyển sang np.frombuffer cho VietOCR
_orig_fromstring = np.fromstring
def _safe_fromstring(string, dtype=float, count=-1, sep=""):
    if sep == "" or sep is None:
        return np.frombuffer(string, dtype=dtype, count=count)
    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
np.fromstring = _safe_fromstring
# ==============================================================================

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    line_dataset_dir = os.path.join(BASE_DIR, "UIT_HWDB_line")
    weights_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Dọn dẹp cache train_data / valid_data tạo dở dang ở lần chạy trước
    for temp_folder in ['train_data', 'valid_data']:
        p = os.path.join(BASE_DIR, temp_folder)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    print("=== BẮT ĐẦU CHUẨN BỊ DỮ LIỆU LINE ===")
    all_samples = []
    ann_file = os.path.join(line_dataset_dir, "train_annotation.txt")
    
    if not os.path.exists(ann_file):
        print(f"Lỗi: Không tìm thấy file {ann_file}")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                img_path = os.path.join(line_dataset_dir, parts[0])
                if os.path.exists(img_path):
                    all_samples.append(f"{parts[0]}\t{parts[1]}")

    random.seed(42)
    random.shuffle(all_samples)
    val_samples = all_samples[:500]
    train_samples = all_samples[500:]

    train_ann_path = os.path.join(BASE_DIR, "train_line_ann.txt")
    val_ann_path = os.path.join(BASE_DIR, "val_line_ann.txt")

    with open(train_ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_samples))
    with open(val_ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_samples))

    print(f"-> Train: {len(train_samples)} mẫu | Val: {len(val_samples)} mẫu")

    # Cấu hình VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')
    config['dataset']['data_root'] = line_dataset_dir
    config['dataset']['train_annotation'] = train_ann_path
    config['dataset']['valid_annotation'] = val_ann_path

    config['trainer']['export'] = os.path.join(weights_dir, 'vietocr_hw_best.pth')
    config['trainer']['checkpoint'] = os.path.join(weights_dir, 'checkpoint_line.pth')
    config['trainer']['log'] = os.path.join(weights_dir, 'train_line.log')

    # Siêu tham số
    config['trainer']['batch_size'] = 16
    config['trainer']['print_every'] = 100
    config['trainer']['valid_every'] = 400
    config['trainer']['iters'] = 6000
    config['trainer']['lr'] = 1e-4
    config['dataloader']['num_workers'] = 0  # Chạy đơn luồng an toàn tuyệt đối trên Windows
    config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config.save(os.path.join(weights_dir, 'config_line.yml'))

    print(f"=== BẮT ĐẦU KHỞI CHẠY (Device: {config['device']}) ===")
    trainer = Trainer(config=config, pretrained=True)
    trainer.train()

if __name__ == '__main__':
    main()