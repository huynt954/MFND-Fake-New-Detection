# Nhận diện tin giả đa phương tiện (Two-Stream Fusion)

Thực nghiệm đi kèm tiểu luận môn Nhập môn Xử lý ngôn ngữ tự nhiên – bài toán **Multimodal Fake News Detection (MFND)**.
Mô hình: **BERT** (văn bản) + **ResNet-50** (hình ảnh) + **chú ý chéo (Cross-Attention)** + Dense/Softmax; tùy chọn nối chữ trong ảnh bằng **PaddleOCR**. Nhãn: `0 = Fake`, `1 = Real`.

Sinh viên: Nguyễn Thanh Huy – TTNT 47

## Chạy trên Google Colab
Mở notebook: https://colab.research.google.com/github/YOUR_USERNAME/REPO/blob/main/notebooks/MFND_Colab.ipynb
(thay `YOUR_USERNAME/REPO` bằng tên tài khoản và tên repo của bạn), chọn *Runtime → Change runtime type → T4 GPU*, rồi chạy lần lượt từng ô.

## Cấu trúc thư mục
| Tệp | Nội dung |
|---|---|
| `model.py` | Mô hình `TwoStreamMFND` (các chế độ: `cross`, `concat`, `text_only`, `image_only`) |
| `data.py` | Đọc CSV, tokenizer, tiền xử lý ảnh |
| `prepare_weibo.py` | Chuyển bộ Weibo gốc sang `data.csv`, chia 80/10/10 theo nhóm ảnh |
| `ocr.py` | Chạy PaddleOCR một lần, lưu cột `ocr_text` |
| `train.py` | Huấn luyện, dừng sớm, đánh giá trên tập kiểm tra |
| `notebooks/MFND_Colab.ipynb` | Notebook chạy toàn bộ quy trình |
| `results/` | Kết quả (`results.json`, bảng tổng hợp) |

## Dữ liệu
Bộ dữ liệu **Weibo** (Jin và cộng sự, ACM Multimedia 2017), gồm tin tức mạng xã hội kèm hình ảnh và nhãn tin thật/tin giả. **Dữ liệu không được đưa vào repo này** (dung lượng lớn, cần tải từ nguồn gốc). Hãy trích dẫn bài báo gốc khi sử dụng:
Z. Jin, J. Cao, H. Guo, Y. Zhang, J. Luo, "Multimodal fusion with recurrent neural networks for rumor detection on microblogs," ACM Multimedia, 2017.

## Chạy nhanh bằng dòng lệnh
```bash
pip install -r requirements.txt
python train.py --smoke                      # kiểm tra code (dữ liệu giả, kết quả vô nghĩa)
python prepare_weibo.py --root <thư mục Weibo> --out data/data.csv
python train.py --csv data/data.csv --image_root <thư mục Weibo> --fusion text_only --epochs 3 --out runs/A4_s42
```
Các cấu hình trong nghiên cứu cắt bỏ: `--fusion cross --use_ocr` (A1), `--fusion cross` (A2), `--fusion concat --use_ocr` (A3), `--fusion text_only` (A4), `--fusion image_only` (A5), `--freeze_cnn_epochs 0` (A7).

## Kết quả
*(Điền sau khi chạy. Chỉ ghi số đo được thật; ghi rõ số epoch, hạt giống, có/không dùng OCR.)*

| Cấu hình | Epoch | Seed | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) |
|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … |

## Hạn chế
- Tập Weibo là tiếng Trung; kết quả chưa suy rộng được cho tiếng Việt.
- Thực nghiệm giới hạn về thời gian (ít epoch, ít hạt giống); chưa dùng ViT (A6).
- Mã chỉ là cài đặt tham khảo của tiểu luận, chưa được tối ưu siêu tham số.
