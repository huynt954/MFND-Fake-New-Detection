# Nhận diện tin giả đa phương tiện (Two-Stream Fusion)

Thực nghiệm đi kèm tiểu luận môn Nhập môn Xử lý ngôn ngữ tự nhiên – bài toán **Multimodal Fake News Detection (MFND)**.
Mô hình đề xuất: **BERT** (văn bản) + **ResNet-50** (hình ảnh) + **chú ý chéo (Cross-Attention)** + Dense/Softmax; tùy chọn nối chữ trong ảnh bằng **PaddleOCR**. Nhãn: `0 = Fake`, `1 = Real`.

Sinh viên: Nguyễn Thanh Huy – TTNT 47

## Chạy trên Google Colab
Mở notebook: https://colab.research.google.com/github/huynt954/MFND-Fake-New-Detection/blob/main/MFND_Colab.ipynb
Chọn *Runtime → Change runtime type → T4 GPU*, rồi chạy lần lượt từng ô.

## Cấu trúc
| Tệp | Nội dung |
|---|---|
| `model.py` | Mô hình `TwoStreamMFND` (các chế độ: `cross`, `concat`, `text_only`, `image_only`) |
| `data.py` | Đọc CSV, tokenizer, tiền xử lý ảnh |
| `prepare_weibo.py` | Chuyển bộ Weibo gốc sang `data.csv`, chia 80/10/10 |
| `ocr.py` | Chạy PaddleOCR một lần, lưu cột `ocr_text` |
| `train.py` | Huấn luyện, dừng sớm, đánh giá trên tập kiểm tra |
| `MFND_Colab.ipynb` | Notebook chạy toàn bộ quy trình |
| `bang10_ket_qua.csv`, `runs/` | Kết quả thực nghiệm (`results.json`) |

## Dữ liệu
Bộ dữ liệu **Weibo** (Jin và cộng sự, ACM Multimedia 2017). Dữ liệu **không** được đưa vào repo. Lần chạy này chỉ dùng 4 file văn bản (`train_rumor.txt`, `train_nonrumor.txt`, `test_rumor.txt`, `test_nonrumor.txt`); **chưa có hình ảnh**.
Z. Jin, J. Cao, H. Guo, Y. Zhang, J. Luo, "Multimodal fusion with recurrent neural networks for rumor detection on microblogs," ACM Multimedia, 2017.

## Kết quả thực nghiệm (tập kiểm tra, %)
Cấu hình A4: BERT chỉ văn bản (`bert-base-chinese`), 3 epoch, batch size 32, AdamW (lr 2e-5 cho BERT, 1e-4 cho lớp mới), độ dài tối đa 128, hạt giống 42, GPU T4 trên Colab. Dữ liệu 9,214 bài chia 7,372 / 921 / 921 (huấn luyện / kiểm định / kiểm tra). Lớp dương cho Precision/Recall/F1 là **Fake**.

| Cấu hình | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) |
|---|---|---|---|---|
| A4. Chỉ văn bản | 96.20 | 97.44 | 94.57 | 95.98 |
| A1, A2, A3, A5 | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

F1 (Fake) trên tập kiểm định theo epoch: 0.9220, 0.9205, 0.9327.

## Lưu ý khi diễn giải
- Chia dữ liệu **ngẫu nhiên theo từng bài đăng**, không tách theo sự kiện; các bài cùng sự kiện hoặc gần trùng có thể nằm ở cả tập huấn luyện và kiểm tra nên kết quả có thể cao hơn thực tế.
- Mới chạy **một hạt giống**, chưa có độ lệch chuẩn.
- Chưa dùng hình ảnh nên **chưa kết luận được** về đóng góp của nhánh thị giác, OCR và chú ý chéo. Chưa chạy ViT (A6).
- Tập Weibo là tiếng Trung, chưa suy rộng được cho tiếng Việt.

## Chạy bằng dòng lệnh
```bash
pip install -r requirements.txt
python train.py --smoke                      # kiểm tra code (dữ liệu giả, kết quả vô nghĩa)
python prepare_weibo.py --root <thư mục Weibo> --out data/data.csv --text_only
python train.py --csv data/data.csv --fusion text_only --epochs 3 --out runs/A4_chi_van_ban_s42
```
Các cấu hình khác (cần có ảnh): `--fusion cross --use_ocr` (A1), `--fusion cross` (A2), `--fusion concat --use_ocr` (A3), `--fusion image_only` (A5), `--freeze_cnn_epochs 0` (A7).
