# Phát hiện bạo lực — CNN + BiLSTM + Attention

## Webcam kết hợp hai model

Camera dùng `checkpoints/mobilenet_bilstm_attention.pt` và model cũ
`checkpoints/resnet50_bilstm_attention.pt` (dự phòng `best2.pt`). Điểm bạo lực
là trung bình 50/50 của hai model, với ngưỡng của MobileNet (hiện tại 0.5).
Bộ lọc chuyển động và MoViNet hiện có tiếp tục xử lý điểm kết hợp.
Trang upload giữ nguyên model cũ. Khởi động lại backend để nạp thay đổi.

`/health` trả đường dẫn hai checkpoint và `webcam_ensemble`.
`/predict/live` trả `model_probs.primary` (MobileNet) và
`model_probs.secondary` (model cũ) trước bộ lọc. Hai lượt suy luận làm tăng
độ trễ; cần đo trên webcam thực tế để chọn trọng số và ngưỡng.

Dự án dùng **notebook Kaggle để train** và **FastAPI để chạy web localhost**.
Mô hình phân loại cả clip thành `non_violence` hoặc `violence`.

## 1. Cài đặt và chạy web

Cần Python 3.10+ và checkpoint PyTorch tương thích.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

Có thể chạy `run_web.bat` sau khi kích hoạt môi trường Python.
Trên Windows nên giữ `--loop asyncio:SelectorEventLoop`: vòng lặp mặc định (Proactor) in lỗi
`WinError 10054` và có thể ngừng nhận kết nối khi trình duyệt ngắt request (tắt camera, tải lại trang).

- Upload: `http://127.0.0.1:8000/`
- Webcam: `http://127.0.0.1:8000/live`
- Lịch sử: `http://127.0.0.1:8000/history`
- Trạng thái model: `http://127.0.0.1:8000/health`

Web đọc `checkpoints/best.pt` khi khởi động. Sau khi thay checkpoint, cần restart web.
Kiến trúc và ngưỡng trong checkpoint được ưu tiên so với `configs/default.yaml`.

## 2. Train trên Kaggle

Mở `notebooks/kaggle_train_cnn_lstm.ipynb`. Notebook chứa model, tiền xử lý,
huấn luyện, chọn ngưỡng và đánh giá; không cần upload thư mục `src` lên Kaggle.

1. Add Input AIRTLab với thư mục `non-violent/cam1`, `non-violent/cam2`, `violent/cam1`, `violent/cam2`.
2. Giữ `DATA_MODE='airtlab'`, kiểm tra `DATA_ROOT` đã điền; đặt `INIT_CHECKPOINT` nếu fine-tune từ checkpoint cũ.
3. Bật GPU và chạy các cell theo thứ tự.
4. Tải `best.pt` cùng báo cáo trong thư mục Output `webcam_cnn_lstm`.
5. Chép checkpoint về `checkpoints/best.pt` rồi restart web.

Chế độ AIRTLab tự tạo `manifest.csv` trong Output và chia theo cặp camera cùng tình huống.
Nếu một góc quay ngắn hơn 2,25 giây, cả cặp bị loại và ghi vào `excluded_airtlab.csv`.
Các số liệu ở chế độ này là kết quả **AIRTLab/external**, chưa phải độ chính xác webcam thật.
Đây là chia theo tình huống, không bảo đảm độc lập theo người vì thiếu ID diễn viên/buổi quay.

Khi có video webcam, đổi `DATA_MODE='manifest'` và chỉnh `MANIFEST_PATH` tới CSV của bạn.
Manifest có các cột:

```csv
path,label,group_id,split,domain
videos/session01_normal.mp4,0,session01,train,webcam
videos/session01_fight.mp4,1,session01,train,webcam
```

Đây chỉ là ví dụ định dạng. Cần đủ hai lớp cho cả train, validation và test webcam.

- Nhãn: `0 = non_violence`, `1 = violence`.
- `path` tương đối với thư mục chứa manifest.
- `group_id` gộp các clip cùng người/buổi quay/video nguồn để tránh lẫn giữa các tập.
- `split`: `train`, `val`, `test`.
- `domain`: `webcam` hoặc `external` cho dữ liệu ngoài như RLVS.

Video cần có nhãn nhất quán theo đoạn, FPS hợp lệ và dài ít nhất 2,25 giây.
Notebook lấy 16 frame cách nhau 150 ms, chọn checkpoint và ngưỡng bằng validation của domain tương ứng,
sau đó đánh giá riêng trên test. Dữ liệu gốc tại `data/raw` được giữ để chuẩn bị Input khi cần;
notebook không tự đọc thư mục này. Manifest webcam cần chuẩn bị thủ công.

## 3. Quay video và kiểm tra

Quay một clip webcam:

```powershell
python record_webcam.py --output data/webcam_tests/test01.mp4 --seconds 10
```

Kiểm tra model và notebook bằng bộ kiểm thử offline:

```powershell
python -m unittest discover -s tests -p test_kaggle_notebook.py -v
```

Video tổng hợp trong kiểm thử chỉ dùng để xác minh code, không đo độ chính xác thực tế.

Có thể suy luận file bằng CLI:

```powershell
python -m src.infer --video data/webcam_tests/test01.mp4 --checkpoint checkpoints/best.pt
```

**Giới hạn hiện tại:** đường upload/CLI lấy cửa sổ frame liên tiếp, còn notebook webcam
lấy frame theo thời gian. Kết quả hai luồng chưa thể so trực tiếp. Webcam lấy khoảng mỗi
150 ms, giữ 16 frame; nên đánh giá sau khi buffer đầy. Frontend vẫn cho request từ 8 frame
ở giai đoạn khởi động. Khung HOG trên giao diện là khung người hoặc khung mặc định,
không xác định riêng người nào đang bạo lực.

## 4. Fine-tune cho webcam

Model gốc (`best2.pt`) học trên RWF-2000 (camera an ninh). Để webcam dự đoán đúng,
fine-tune thêm bằng clip quay từ chính webcam:

1. **Thu thập:** mở trang Camera, bật camera, dùng khung "Thu thập dữ liệu webcam".
   Mỗi lần bấm quay clip 4 giây (đếm ngược 3 giây), lưu vào
   `data/webcam_dataset/violence/` hoặc `data/webcam_dataset/non_violence/`.
   Nên có 100–200 clip mỗi lớp; lớp không bạo lực cần nhiều cử động mạnh
   (ôm, nhảy, tập thể dục, vẫy tay, đi lại) và cả cảnh phòng trống.
2. **Chia tập:** tự động theo tên file (~70% train, 10% val, 20% test), xem
   `src/data/webcam_dataset.py`. Không cần tạo manifest.
3. **Đánh giá trước fine-tune:**
   `python -m src.eval_webcam --checkpoint checkpoints/best2.pt --threshold 0.75`
4. **Fine-tune** trên Kaggle/Colab (GPU): upload `data/webcam_dataset` và `best2.pt`,
   sửa `WEBCAM_ROOT`, `INIT_CHECKPOINT` và `EXTRA_DATASETS` (RWF-2000, RLVS, AIRTLab — các bộ `best2.pt` đã học, để model không quên) trong
   `notebooks/finetune_webcam.py` rồi chạy. Chép `best_webcam.pt` về `checkpoints/`.
5. **Đánh giá sau fine-tune:**
   `python -m src.eval_webcam --checkpoint checkpoints/best2.pt checkpoints/best_webcam.pt`
6. Restart web. Trang Camera tự dùng `checkpoints/best_webcam.pt` nếu có, với ngưỡng
   đã chọn trên tập val; trang Upload vẫn dùng `best2.pt`.

## 5. Cấu trúc đang sử dụng

```text
notebooks/kaggle_train_cnn_lstm.ipynb   Train và đánh giá trên Kaggle
src/api.py                           Trang web và API
src/infer.py                         Nạp checkpoint, suy luận video/frame
src/history.py                       SQLite và lưu ảnh/video lịch sử
src/models/cnn_lstm.py                ResNet18/50 + BiLSTM + Attention
src/data/preprocess.py               Chia cửa sổ video cho suy luận
src/data/transforms.py               Resize/crop và chuẩn hóa frame
src/data/webcam_dataset.py           Đọc và chia tập dữ liệu webcam
src/eval_webcam.py                   Đánh giá checkpoint trên clip webcam
notebooks/finetune_webcam.py         Fine-tune best2.pt bằng dữ liệu webcam
src/utils/config.py                  Đọc cấu hình và đường dẫn project
src/utils/viz.py                     Vẽ attention cho video upload
src/smoke_test.py                     Kiểm tra forward model không cần dataset
web/                                 Giao diện upload, webcam, lịch sử
configs/default.yaml                 Cấu hình dự phòng cho backend
checkpoints/best.pt                   Model web hiện sử dụng
data/raw/                            Video nguồn để chuẩn bị dataset
data/webcam_tests/                    Video webcam để kiểm tra
data/history.db                      Lịch sử nhận diện
storage/                             Video/ảnh được lịch sử sử dụng
reports/outputs/                      Kết quả suy luận
tests/test_kaggle_notebook.py          Kiểm thử notebook và tương thích backend
record_webcam.py                      Công cụ quay webcam
run_web.bat                          Khởi động web trên Windows
```

Luồng train local/pipeline demo cũ đã được gỡ; huấn luyện và đánh giá dataset thực hiện trong notebook Kaggle.

## Email cảnh báo cho Upload và Webcam

Đặt `EMAIL_ALERTS_ENABLED=false` trong `.env` để tạm tắt email cho cả Upload và Webcam; đặt `true` để bật lại. Khởi động lại backend sau khi thay đổi.

Cấu hình `ALERT_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` trong `.env` theo `.env.example`, rồi khởi động lại backend. SMTP dùng STARTTLS (mặc định cổng 587). Với Gmail, dùng mật khẩu ứng dụng cho tài khoản gửi.

- Upload: gửi một email sau khi lưu kết quả nếu nhãn toàn video là bạo lực hoặc có đoạn bạo lực, kể cả khi nhãn toàn video không bạo lực.
- Webcam: gửi một email khi sự kiện kết thúc và đã lưu lịch sử; không gửi theo từng frame hoặc lượt dự đoán.
- Giao diện hiển thị trạng thái gửi mail. Lỗi SMTP không làm mất kết quả phân tích hoặc bản ghi đã lưu.
- Kiểm thử không gửi email thật: `python -m unittest discover -s tests -p test_mail_alert.py -v`.

## Xuất báo cáo PDF

Trang Upload và Webcam tải trực tiếp file `.pdf` qua `/reports/pdf`, không cần in từ HTML. Backend dùng Matplotlib và font DejaVu Sans hỗ trợ tiếng Việt. Khởi động lại backend sau khi cập nhật source.
