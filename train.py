"""Huấn luyện + đánh giá mô hình (tiểu luận mục 3.7 và 4.2 – 4.6).
Lớp dương cho Precision/Recall/F1 là Fake (nhãn 0), đúng như Bảng 7 trong tiểu luận.

Ví dụ:
  python train.py --smoke                                  # chạy thử nhanh, không cần dữ liệu/mạng
  python train.py --csv data_ocr.csv --image_root imgs --fusion cross --use_ocr --out runs/A1
"""
import argparse, copy, json, os, random, tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data import load_splits, MFNDDataset
from model import TwoStreamMFND


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_smoke(tmp):
    """Dữ liệu giả + tokenizer tối thiểu để kiểm tra code chạy được (kết quả KHÔNG có ý nghĩa)."""
    from PIL import Image
    from transformers import BertTokenizerFast
    letters = list("abcdefghijklmnopqrstuvwxyz")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + letters + ["##" + c for c in letters]
    vp = os.path.join(tmp, "vocab.txt"); open(vp, "w").write("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=vp, do_lower_case=True)
    rng = np.random.RandomState(0); rows = []
    for i in range(48):
        Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(os.path.join(tmp, f"{i}.png"))
        words = " ".join("".join(rng.choice(letters, 4)) for _ in range(6))
        rows.append({"text": words, "ocr_text": "ab cd", "image": f"{i}.png", "label": i % 2})
    csv = os.path.join(tmp, "smoke.csv"); pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, tok, len(vocab)


@torch.no_grad()
def evaluate(model, loader, device, need_images):
    model.eval(); ys, ps = [], []
    for b in loader:
        img = b["image"].to(device) if need_images else None
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), img)
        ps += logits.argmax(-1).tolist(); ys += b["label"].tolist()
    p, r, f, _ = precision_recall_fscore_support(ys, ps, labels=[0], average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(ys, ps)), "precision_fake": float(p[0]),
            "recall_fake": float(r[0]), "f1_fake": float(f[0]),
            # hàng = thực tế [Fake, Real], cột = dự đoán [Fake, Real] -> [[TP, FN], [FP, TN]]
            "confusion": confusion_matrix(ys, ps, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--image_root", default="")
    ap.add_argument("--text_model", default="bert-base-chinese")
    ap.add_argument("--fusion", default="cross", choices=["cross", "concat", "text_only", "image_only"])
    ap.add_argument("--use_ocr", action="store_true")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr_bert", type=float, default=2e-5); ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--lr_cnn", type=float, default=1e-5); ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--freeze_cnn_epochs", type=int, default=2)  # A7: đặt 0 để không đóng băng dần
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="runs/exp"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiny_vocab = None
    if a.smoke:
        tmp = tempfile.mkdtemp(); a.csv, tok, tiny_vocab = make_smoke(tmp)
        a.image_root, a.use_ocr, a.epochs, a.batch_size, a.workers, a.freeze_cnn_epochs = tmp, True, 2, 8, 0, 1
    else:
        assert a.csv, "Cần --csv (hoặc dùng --smoke)"
        tok = AutoTokenizer.from_pretrained(a.text_model)
    need_images = a.fusion != "text_only"
    tr, va, te = load_splits(a.csv, a.seed)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)} | device = {device}")

    def loader(df, train):
        ds = MFNDDataset(df, tok, train, a.max_len, a.use_ocr, a.image_root, need_images)
        return DataLoader(ds, a.batch_size, shuffle=train, num_workers=a.workers)
    tl, vl, sl = loader(tr, True), loader(va, False), loader(te, False)

    model = TwoStreamMFND(a.text_model, a.fusion, pretrained_image=not a.smoke, tiny_vocab=tiny_vocab).to(device)
    groups, seen = [], set()
    def add(params, lr):
        params = [p for p in params if id(p) not in seen]
        seen.update(id(p) for p in params)
        if params: groups.append({"params": params, "lr": lr, "weight_decay": a.weight_decay})
    if model.text is not None: add(model.text.parameters(), a.lr_bert)
    if model.image is not None: add(model.image.backbone.parameters(), a.lr_cnn)
    add(model.parameters(), a.lr_new)  # lớp mới: proj, cross-attention, head
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * len(tl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)  # warm-up 10% rồi giảm tuyến tính
    weight = None
    if a.class_weight:
        n = tr.label.value_counts().sort_index().values
        weight = torch.tensor(len(tr) / (2.0 * n), dtype=torch.float).to(device)
    lossf = nn.CrossEntropyLoss(weight=weight)  # Softmax + Cross-Entropy

    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        frozen = model.image is not None and ep <= a.freeze_cnn_epochs
        if model.image is not None:
            for p in model.image.backbone.parameters(): p.requires_grad = not frozen
        model.train()
        if frozen: model.image.backbone.eval()
        tot = 0.0
        for b in tl:
            img = b["image"].to(device) if need_images else None
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), img)
            loss = lossf(logits, b["label"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        m = evaluate(model, vl, device, need_images)
        hist.append({"epoch": ep, "train_loss": tot / len(tl), **m})
        print(f"epoch {ep}: loss={tot/len(tl):.4f} val F1(Fake)={m['f1_fake']:.4f} acc={m['accuracy']:.4f}")
        if m["f1_fake"] > best:
            best, bad, best_state = m["f1_fake"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience: print("Dừng sớm (early stopping)"); break
    model.load_state_dict(best_state)
    test = evaluate(model, sl, device, need_images)  # chỉ đo trên tập test MỘT LẦN, sau khi chốt mô hình
    print("TEST:", json.dumps(test))
    json.dump({"args": vars(a), "best_val_f1_fake": best, "test": test, "history": hist},
              open(os.path.join(a.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
