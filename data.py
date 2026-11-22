"""Đọc dữ liệu từ file CSV có các cột: text, image, label (0=Fake, 1=Real);
tùy chọn: ocr_text (do ocr.py tạo), split (train/val/test)."""
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # chuẩn hóa theo ImageNet


def build_transform(train):
    if train:  # tăng cường nhẹ, KHÔNG lật ngang (tránh đảo chữ trong ảnh)
        ops = [transforms.Resize((256, 256)), transforms.RandomCrop(224),
               transforms.ColorJitter(brightness=0.2)]
    else:
        ops = [transforms.Resize((224, 224))]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_splits(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    for c in ("text", "label"):  # cột 'image' chỉ cần khi dùng ảnh (không cần với --fusion text_only)
        assert c in df.columns, f"CSV thiếu cột '{c}'"
    if "split" in df.columns:
        return (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print("[cảnh báo] CSV không có cột 'split' -> chia phân tầng 80/10/10 ngẫu nhiên. "
          "Nếu nhiều bài dùng chung ảnh/sự kiện, hãy tự tạo cột 'split' để tránh rò rỉ dữ liệu.")
    train, rest = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class MFNDDataset(Dataset):
    def __init__(self, df, tokenizer, train, max_len=128, use_ocr=False, image_root="", need_images=True):
        self.df, self.tok, self.max_len = df, tokenizer, max_len
        self.use_ocr, self.root, self.need_images = use_ocr, image_root, need_images
        self.tf = build_transform(train)
        if use_ocr:
            assert "ocr_text" in df.columns, "Thiếu cột ocr_text: chạy ocr.py trước"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = str(r["text"]) if pd.notna(r["text"]) else ""
        pair = None
        if self.use_ocr and pd.notna(r["ocr_text"]) and str(r["ocr_text"]).strip():
            pair = str(r["ocr_text"])  # [CLS] văn bản [SEP] chữ OCR [SEP]
        enc = self.tok(text, pair, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(r["label"]), dtype=torch.long)}
        if self.need_images:
            try:
                img = Image.open(os.path.join(self.root, str(r["image"]))).convert("RGB")
                item["image"] = self.tf(img)
            except Exception:  # ảnh hỏng/thiếu -> ảnh trống
                item["image"] = torch.zeros(3, 224, 224)
        return item
