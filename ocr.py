"""Chạy PaddleOCR một lần cho toàn bộ ảnh và lưu vào cột ocr_text (bộ nhớ đệm).
Dùng: python ocr.py --csv data.csv --image_root imgs --out data_ocr.csv --lang ch
LƯU Ý: API PaddleOCR đổi nhiều giữa các phiên bản (2.x và 3.x); file này thử cả hai.
Nếu lỗi, gửi lại thông báo lỗi + phiên bản paddleocr để chỉnh."""
import argparse, os
import pandas as pd


def make_ocr(lang):
    from paddleocr import PaddleOCR
    try:
        return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)  # 2.x
    except (TypeError, ValueError):
        return PaddleOCR(lang=lang)                                      # 3.x


def read_lines(ocr, path):
    """Trả về danh sách (y, x, text, conf)."""
    out = []
    if hasattr(ocr, "predict"):  # 3.x
        for r in ocr.predict(path):
            texts, scores = r["rec_texts"], r["rec_scores"]
            boxes = r.get("rec_polys", r.get("dt_polys"))
            for t, s, b in zip(texts, scores, boxes):
                out.append((float(min(p[1] for p in b)), float(min(p[0] for p in b)), t, float(s)))
    else:                        # 2.x
        res = ocr.ocr(path, cls=True)
        for box, (t, s) in (res[0] or []):
            out.append((min(p[1] for p in box), min(p[0] for p in box), t, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--image_root", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--lang", default="ch")
    ap.add_argument("--min_conf", type=float, default=0.6)  # bỏ dòng có độ tin cậy thấp
    a = ap.parse_args()
    df = pd.read_csv(a.csv); ocr = make_ocr(a.lang); texts = []
    for i, p in enumerate(df["image"]):
        try:
            lines = [l for l in read_lines(ocr, os.path.join(a.image_root, str(p))) if l[3] >= a.min_conf]
            lines.sort(key=lambda l: (l[0], l[1]))  # thứ tự đọc: trên xuống, trái sang
            texts.append(" ".join(l[2] for l in lines))
        except Exception as e:
            print("OCR lỗi:", p, e); texts.append("")
        if i % 200 == 0: print(f"{i}/{len(df)}")
    df["ocr_text"] = texts
    df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
