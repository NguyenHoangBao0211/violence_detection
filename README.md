# Phát hiện bạo lực — CNN + BiLSTM + Attention

Dự án dùng **notebook Kaggle để train** và **FastAPI để chạy web localhost**.
Mô hình phân loại cả clip thành `non_violence` hoặc `violence`.

## 1. Cài đặt và chạy web

Cần Python 3.10+ và checkpoint PyTorch tương thích.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Có thể chạy `run_web.bat` sau khi kích hoạt môi trường Python.

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

## 4. Cấu trúc đang sử dụng

```text
notebooks/kaggle_train_cnn_lstm.ipynb   Train và đánh giá trên Kaggle
src/api.py                           Trang web và API
src/infer.py                         Nạp checkpoint, suy luận video/frame
src/history.py                       SQLite và lưu ảnh/video lịch sử
src/models/cnn_lstm.py                ResNet18/50 + BiLSTM + Attention
src/data/preprocess.py               Chia cửa sổ video cho suy luận
src/data/transforms.py               Resize/crop và chuẩn hóa frame
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
