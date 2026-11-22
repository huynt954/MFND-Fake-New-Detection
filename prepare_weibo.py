"""Chuyển bộ Weibo (Jin và cộng sự, 2017) sang data.csv cho train.py.

Cấu trúc mong đợi (theo mô tả của bộ dữ liệu): mỗi bài đăng gồm 3 dòng:
  dòng 1: 15 trường meta ngăn bằng '|' (trường đầu là id bài đăng)
  dòng 2: URL các ảnh ngăn bằng '|' (hoặc 'null')
  dòng 3: nội dung văn bản
Các file *nonrumor*.txt -> Real (nhãn 1); file *rumor*.txt còn lại -> Fake (nhãn 0).
Ảnh nằm trong các thư mục ảnh (tên file ảnh = phần cuối của URL).

Dùng:
  python prepare_weibo.py --root /content/weibo --out /content/data/data.csv            # văn bản + ảnh
  python prepare_weibo.py --root /content/weibo --out /content/data/data.csv --text_only  # chỉ văn bản (chưa có ảnh)
Chia 80/10/10 THEO NHÓM ảnh (các bài dùng chung một ảnh luôn nằm cùng một tập) để tránh rò rỉ dữ liệu.
Nếu cấu trúc file thật khác, hãy gửi lại vài dòng đầu của một file txt để chỉnh."""
import argparse, os, random, re
import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def parse(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n").rstrip("\r") for l in f]
    if len(lines) % 3 != 0:
        print(f"[cảnh báo] {path}: số dòng ({len(lines)}) không chia hết cho 3, bỏ phần dư")
    return [(lines[i].split("|")[0], lines[i + 1], lines[i + 2]) for i in range(0, len(lines) - 2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--text_only", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    txts, imgs = [], {}
    for d, _, fs in os.walk(a.root):
        for f in fs:
            fl = f.lower()
            if fl.endswith(".txt") and "rumor" in fl:
                txts.append(os.path.join(d, f))
            elif fl.endswith(IMG_EXT) and not a.text_only:
                imgs.setdefault(f, os.path.relpath(os.path.join(d, f), a.root))
    assert txts, "Không tìm thấy file *rumor*.txt trong --root"
    print("Tìm thấy file:", [os.path.basename(t) for t in txts], "| số ảnh:", len(imgs))
    rows, skipped = [], 0
    for t in sorted(txts):
        name = os.path.basename(t).lower()
        label = 1 if "nonrumor" in name else 0  # 0 = Fake, 1 = Real
        orig = "test" if "test" in name else "train"
        for pid, urls, text in parse(t):
            text = re.sub(r"\s+", " ", re.sub(r"https?://\S+", " ", text)).strip()
            if not text:
                skipped += 1; continue
            image = ""
            if not a.text_only:
                for u in urls.split("|"):
                    b = u.strip().split("/")[-1]
                    if b in imgs:
                        image = imgs[b]; break
                if not image:
                    skipped += 1; continue
            rows.append({"id": pid, "text": text, "image": image, "label": label, "orig_split": orig})
    df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    df["group"] = df["text"] if a.text_only else df["image"]
    keys = list(df["group"].unique()); random.Random(a.seed).shuffle(keys)
    size = df["group"].value_counts(); n, cum, m = len(df), 0, {}
    for k in keys:
        m[k] = "train" if cum < 0.8 * n else ("val" if cum < 0.9 * n else "test"); cum += size[k]
    df["split"] = df["group"].map(m)
    df.drop(columns="group").to_csv(a.out, index=False)
    print(f"Đã ghi {a.out}: {len(df)} bài (bỏ {skipped} bài thiếu văn bản/ảnh)")
    print(pd.crosstab(df["split"], df["label"].map({0: "Fake", 1: "Real"})))


if __name__ == "__main__":
    main()
