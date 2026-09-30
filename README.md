# Trợ lý Tim Mạch - Phân loại sự hiện diện bệnh tim trong dữ liệu

Project này xây dựng hệ thống AI hỗ trợ phân loại hồ sơ thuộc nhóm có hoặc không có bệnh tim theo bốn nguồn của UCI Heart Disease Dataset, đồng thời phân tích những yếu tố có ảnh hưởng lớn nhất đến kết quả dự đoán.

## Chạy từ GitHub

Trong repository GitHub, mã nguồn của project nằm ở thư mục `heart-disease-ai`. Sau khi clone, hãy mở terminal tại thư mục này trước khi chạy các lệnh bên dưới. Dữ liệu, model, biểu đồ và báo cáo được tạo lại tự động nên không được lưu trong Git.

## Mục tiêu để nhắm bài 10 điểm

- Có dữ liệu thật từ UCI Heart Disease Dataset.
- Có tiền xử lý đúng chuẩn bằng đường ống xử lý để tránh rò rỉ dữ liệu.
- Có so sánh nhiều mô hình: Hồi quy Logistic, K láng giềng gần nhất, Máy vector hỗ trợ, Rừng ngẫu nhiên, Tăng cường Gradient, XGBoost.
- Có tìm kiếm tham số và kiểm định chéo 5 phần trên tập huấn luyện, không chọn mô hình bằng tập kiểm tra.
- Có calibration xác suất bằng `CalibratedClassifierCV`.
- Có tối ưu mốc sàng lọc theo độ nhạy và điểm F2; đây là lựa chọn kỹ thuật của bài, không phải ngưỡng chẩn đoán y khoa.
- Có đánh giá bằng độ chính xác tổng thể, độ chính xác cảnh báo, độ nhạy, độ đặc hiệu, điểm F1, điểm F2, AUC ROC, độ chính xác trung bình, sai số xác suất Brier.
- Có ma trận nhầm lẫn, đường cong ROC, đường cong độ chính xác - độ nhạy, đường hiệu chỉnh xác suất, biểu đồ tối ưu ngưỡng.
- Có giải thích yếu tố ảnh hưởng bằng mức ảnh hưởng theo hoán vị, biểu đồ xu hướng xác suất theo từng yếu tố và SHAP khi mô hình hỗ trợ trực tiếp.
- Có giao diện Streamlit cho người dùng nhập hồ sơ, xem xác suất dự đoán, khuyến nghị theo mốc sàng lọc và lý do dự đoán.
- Có notebook và test tự động để chứng minh quy trình làm bài.

## Kết quả hiện tại

- Mô hình được chọn bằng kiểm định chéo: `K láng giềng gần nhất`.
- Mô hình triển khai: `K láng giềng gần nhất đã hiệu chỉnh xác suất`.
- Ngưỡng quyết định: `0.290`.
- Kết quả tập kiểm tra sau hiệu chỉnh xác suất và tối ưu ngưỡng: độ chính xác tổng thể `0.788`, độ chính xác cảnh báo `0.733`, độ nhạy `0.971`, độ đặc hiệu `0.561`, AUC ROC `0.880`, sai số xác suất Brier `0.139`.
- Dữ liệu huấn luyện: `920` hồ sơ từ Cleveland, Hungary, Switzerland và VA Long Beach; còn `9` biến đầu vào sau bước lọc.
- Tiêu chí chọn mô hình: `0.45 * độ nhạy CV + 0.35 * AUC ROC CV + 0.20 * F1 CV`.

## Cấu trúc thư mục

```text
D:\BTL.benhtim
|-- app.py
|-- requirements.txt
|-- data/
|   |-- heart_uci_expanded_raw.csv
|   |-- heart_uci_expanded_processed.csv
|   |-- uci_*.data                 (bản tải từ bốn nguồn UCI)
|   |-- uci_dataset_sources.csv
|   |-- feature_filter_report.csv
|   |-- data_dictionary.csv
|   |-- patient_records.csv       (tự tạo sau khi lưu hồ sơ, không đưa lên Git)
|-- models/
|   |-- best_model.joblib
|   |-- best_raw_model.joblib
|   |-- trained_models.joblib
|   |-- model_metadata.json
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_modeling.ipynb
|-- reports/
|   |-- bao_cao_tom_tat.md
|   |-- model_metrics.csv
|   |-- cv_grid_search_results.csv
|   |-- feature_importance.csv
|   |-- threshold_analysis.csv
|   |-- test_predictions.csv
|   |-- slide_outline.md
|   |-- figures/
|-- src/
|   |-- create_notebooks.py
|   |-- collect_uci_data.py
|   |-- data_utils.py
|   |-- download_data.py
|   |-- predict.py
|   |-- settings.py
|   |-- smoke_test.py
|   |-- train_model.py
```

## Cách chạy

Nếu dùng virtual environment đã tạo trong thư mục project:

```powershell
.\.venv\Scripts\python.exe -m src.collect_uci_data
.\.venv\Scripts\python.exe -m src.train_model
.\.venv\Scripts\python.exe -m src.create_notebooks
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nếu máy đã có Python trong PATH:

```powershell
pip install -r requirements.txt
python -m src.collect_uci_data
python -m src.train_model
python -m src.create_notebooks
python -m unittest discover -s tests
streamlit run app.py
```

## Giao diện demo

Ứng dụng Streamlit được thiết kế theo phong cách y tế: màu nền sáng, thông tin rõ ràng, hạn chế trang trí thừa, có card chỉ số quan trọng, đồng hồ xác suất, khuyến nghị theo mốc sàng lọc và phần giải thích yếu tố ảnh hưởng cho từng hồ sơ.

Các tab chính:

- `Phân tích hồ sơ`: nhập tên bệnh nhân và các chỉ số, xem kết quả, sau đó bấm `Lưu hồ sơ và kết quả`.
- `Hồ sơ đã lưu`: xem lịch sử các lần đánh giá và tải toàn bộ dữ liệu ra CSV.
- `Hiệu năng mô hình` và `Dữ liệu và giải thích`: xem thông tin kỹ thuật của mô hình.

Hồ sơ bệnh nhân được lưu cục bộ tại `data/patient_records.csv`. Đây là cơ chế lưu cho bản demo/học tập, không thay thế hệ thống hồ sơ bệnh án chính thức.


## Phạm vi yếu tố đầu vào

Project hiện dùng 9 yếu tố có thể hài hòa khi đối chiếu bốn nguồn: tuổi, giới tính, kiểu đau ngực, huyết áp lúc nghỉ, cholesterol, điện tâm đồ lúc nghỉ, nhịp tim tối đa, đau thắt ngực khi vận động và thay đổi đoạn ST.

Đã loại khỏi đầu vào mô hình: `fbs`, `slope`, `ca`, `thal`. Các biến này bị thiếu quá nhiều ở ít nhất một nguồn nên không còn xuất hiện trong giao diện dự đoán. Báo cáo chi tiết nằm tại `data/feature_filter_report.csv`.

Đây là đủ và đúng với đề bài ở phạm vi học máy. Khi thuyết trình cần nói rõ hệ thống chưa bao gồm toàn bộ yếu tố y khoa ngoài đời như hút thuốc, BMI, tiền sử gia đình, thuốc đang dùng hoặc bệnh nền khác.

## Nguồn dữ liệu

- Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), gồm các nguồn Cleveland, Hungary, Switzerland và VA Long Beach.
- Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Bộ dữ liệu sau khi gộp có 920 mẫu và 9 đặc trưng đầu vào. Nhãn gốc `target` có giá trị `0-4`; project chuyển thành bài toán nhị phân về sự hiện diện bệnh trong dữ liệu:

```text
0 = không có bệnh theo nhãn dữ liệu
1 = có bệnh theo nhãn dữ liệu
```

## Lưu ý y khoa

Hệ thống chỉ phục vụ mục đích học tập và phân tích hồ sơ tham khảo. Kết quả không phải xác suất mắc bệnh trong tương lai, không thay thế chẩn đoán, tư vấn hoặc điều trị của bác sĩ/cơ sở y tế. Các biến bị loại khỏi mô hình không được tự suy đoán hoặc tự nhập thay thế.
